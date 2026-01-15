import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from logger import get_logger

from torch import optim, nn
from torch.utils.data import DataLoader
from torchinfo import summary

from Model import LSTM, Linear_model, CNN_1d
from Dataset.BCG_Dataset import BCGSynthesisDataset, BCGSynthesisDataset_V3
from Algorithm.ECG_heartrate_alg import ECG_R_peak_weight
from Model.Loss_Function import MorletCWTLoss, MultiResolutionSTFTLoss

# Parameters
BATCH_SIZE = 128
NUM_EPOCHS = 200
NUM_CLASSES = 1
INPUT_LEN = 1000

MOMENTUM = 0.9
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-3
STEP_SIZE = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def stft_loss(pred_bcg, target_bcg, window_length=256, fs=100, band=10, w_time=0.5, w_fft=0.5, eps=1e-7):
    if pred_bcg.dim() == 3:
        pred_bcg = pred_bcg.squeeze(1)
    if target_bcg.dim() == 3:
        target_bcg = target_bcg.squeeze(1)

    window = torch.hann_window(window_length=window_length).to(DEVICE)
    # pred,target: [B,C,T]
    pred_stft = torch.stft(pred_bcg, win_length=window_length, window=window, n_fft=256, hop_length=32, return_complex=True)
    target_stft = torch.stft(target_bcg, win_length=window_length, window=window, n_fft=256, hop_length=32, return_complex=True)

    sc_loss = torch.norm(torch.abs(target_stft) - torch.abs(pred_stft), p='fro')/torch.norm(torch.abs(target_stft), p='fro')
    mag_loss = F.l1_loss(torch.log(torch.abs(target_stft)+1e-1), torch.log(torch.abs(pred_stft)+1e-1))
    mse_time = torch.mean((pred_bcg - target_bcg)**2)


    return w_time*mse_time + (w_fft)*sc_loss + eps #+ (w_fft/2)*mag_loss

def fft_loss(pred_bcg, target_bcg, fs=100, band=10, w_time=0.2, w_fft=0.8, eps=1e-7):
    # pred,target: [B,C,T]
    pred_fft = torch.abs(torch.fft.rfft(pred_bcg, dim=-1))  # [B,C,F]
    target_fft = torch.abs(torch.fft.rfft(target_bcg, dim=-1))

    freqs = torch.fft.rfftfreq(pred_bcg.shape[-1], d=1.0/fs).to(pred_bcg.device)
    band_mask = freqs <= band
    band_mask = band_mask
    pred_fft = pred_fft[:, :, band_mask]
    target_fft = target_fft[:, :, band_mask]

    pred_fft = (pred_fft - pred_fft.mean(dim=-1, keepdim=True)) / pred_fft.std(dim=-1, keepdim=True)
    target_fft = (target_fft - target_fft.mean(dim=-1, keepdim=True)) / target_fft.std(dim=-1, keepdim=True)

    mse_time = torch.mean((pred_bcg - target_bcg)**2)
    mse_fft = torch.mean((pred_fft - target_fft)**2)

    return w_time*mse_time + w_fft*mse_fft + eps

def train(net, device, epochs, lr, train_loader, test_loader, WeightDataPath, logger_record):
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.1)  # 每50個epochs lr乘以0.5
    # cwt_loss = MultiResolutionSTFTLoss().to(DEVICE)
    cwt_loss = MorletCWTLoss(fs=100, fmin=0.7, fmax=10.0, num_freqs=48,
                         kernel_size=401, sigma=0.3, log_compress=True).to(DEVICE)
    fn_loss = nn.BCEWithLogitsLoss()

    # criterion = nn.MSELoss()

    best_train_loss = float('inf')
    best_val_loss = float('inf')

    total_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        net.train()
        running_train_loss = 0.0
        batch_count = 0

        current_lr = optimizer.param_groups[0]['lr']
        logger_record.info(f"[Epoch {epoch+1}/{epochs}] Current LR: {current_lr:.10f}")
        
        # for BCGSignal, label in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
        for BCGSignal, label, conf_label in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            BCGSignal = BCGSignal.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            conf_label = conf_label.to(device, dtype=torch.float32)
            prediction, predict_conf_score = net(BCGSignal)
            
            """ original MSE loss """
            cwt_time = cwt_loss(prediction, label)
            mse_time = torch.mean((prediction - label)**2)
            score_time = fn_loss(predict_conf_score, conf_label)
            loss = cwt_time*0.5 + mse_time*0.5 + score_time
            # loss = stft_loss(prediction, label)
            """ R peak weight MSE loss """
            # with torch.no_grad():
            #     r_mask = batch_detect_rpeaks(label)

            # # 加權 MSE Loss
            # loss = weighted_mse_loss(prediction, label, r_mask)

            """ backward propagation """
            optimizer.zero_grad()
            loss.backward()
            """ avoid gradient explosion """
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)  # ← 加這行
            optimizer.step()

            running_train_loss += loss.item()
            batch_count += 1

            if loss.item() < best_train_loss:
                best_train_loss = loss.item()
                torch.save(net.state_dict(), os.path.join(WeightDataPath, 'best_Train_model.pth'))

        avg_train_loss = running_train_loss / batch_count
        total_loss_list.append(avg_train_loss)
        logger_record.info(f"Epoch {epoch+1}: Avg Train Loss = {avg_train_loss:.6f}")

        # Validation
        net.eval()
        with torch.no_grad():
            val_loss = test(net, test_loader, cwt_loss, fn_loss, device)
            val_loss_list.append(val_loss)
            logger_record.info(f"Epoch {epoch+1}: Validation Loss = {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(net.state_dict(), os.path.join(WeightDataPath, 'best_Test_model.pth'))
                logger_record.info(f"New best validation model saved at epoch {epoch+1}, loss: {best_val_loss:.6f}")

        # Update learning rate
        scheduler.step()

    logger_record.info(f"best validation model loss: {best_val_loss:.6f}")

    return total_loss_list, val_loss_list

def test(net, test_iter, criterion, score_criterion, device):
    total_loss_test = 0
    nums_ = 0

    with torch.no_grad():
        # for X, y in tqdm(test_iter, desc=f"Validation", leave=False):
        for X, y, z in tqdm(test_iter, desc=f"Validation", leave=False):
            X, y, z = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32), z.to(device=device, dtype=torch.float32)
            predict, predict_conf_score = net(X)
            """ original MSE loss """
            # loss = stft_loss(predict, y)
            cwt_time = criterion(predict, y)
            mse_time = torch.mean((predict - y)**2)
            score_time = score_criterion(predict_conf_score, z)
            loss = cwt_time*0.5 + mse_time*0.5 + score_time
            """ r peak weight MSE loss """
            # r_mask = batch_detect_rpeaks(y)
            # loss = weighted_mse_loss(predict, y, r_mask)
            total_loss_test += loss.item()
            nums_ += 1

    total_loss_test /= nums_

    return total_loss_test

weightPath = r'D:\Program\innolux_BCG_deep_learning\weight\BCG_HeartFilter\251231'
logger_record = get_logger(filename=os.path.join(weightPath, 'train_logger.log'))
logger_record.info(f'Using synthesis BCG data to training model: add bad data to let model predict confident score for data')
logger_record.info(f'\n*** parameters ***\nBatch size:{BATCH_SIZE}\nepochs:{NUM_EPOCHS}\nchannels:{NUM_CLASSES}\ndata length:{INPUT_LEN}\
                   \nlearning rate: {LEARNING_RATE}\nDevice:{DEVICE}\nweight decay: {WEIGHT_DECAY}\nlearning rate step size: {STEP_SIZE}')

# trainset = BCGSynthesisDataset(root=r'E:\BCG_innolux_dataset\BCG_synthesis_data\train')
# valset = BCGSynthesisDataset(root=r'E:\BCG_innolux_dataset\BCG_synthesis_data\validation')
trainset = BCGSynthesisDataset_V3(root=r'E:\BCG_innolux_dataset\BCG_synthesis_data_v3\train')
valset = BCGSynthesisDataset_V3(root=r'E:\BCG_innolux_dataset\BCG_synthesis_data_v3\validation')

train_loader = DataLoader(dataset=trainset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=valset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          drop_last=True)

# model = LSTM.LSTM_BCGFilter_Pre(seq_len=INPUT_LEN, input_size=1, hidden_size=128, output_size=1,
#             dropout=0.2, num_layers=6, bidirectional=True, repeat_times = 1)

model = LSTM.LSTM_BCGFilter_Pre_Confidence(seq_len=INPUT_LEN, input_size=1, hidden_size=128, output_size=1,
            dropout=0.2, num_layers=6, bidirectional=True, repeat_times = 1)

# model = Linear_model.LinearBCG(seq_len=1000, p_drop=0.2, hidden=512)
# model = CNN_1d.BCG_CNN_Small(seq_len=1000, channels=1, hidden=32, dropout=0.2)
model.to(DEVICE)
# model.load_state_dict(torch.load(r'weight\BCG_contrustion\251001\best_Test_model.pth', map_location=DEVICE))

summary_output_path = os.path.join(weightPath, 'model_summary.txt')
with open(summary_output_path, "w", encoding='utf-8-sig') as f:
    report = summary(
                model,
                input_size=(1, NUM_CLASSES, INPUT_LEN),
                device=DEVICE  # 每一行寫進檔案
            )
    f.write(str(report))


Total_Loss, Val_Loss = train(model, DEVICE, NUM_EPOCHS, LEARNING_RATE, train_loader, val_loader, weightPath, logger_record)

plt.plot(Total_Loss, label='Train loss')
plt.plot(Val_Loss, label="Val loss")
plt.legend()
plt.savefig(os.path.join(weightPath, 'loss.png'))
plt.close()