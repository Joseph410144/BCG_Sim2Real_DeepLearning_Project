# BCG Cardiac Representation Research

本專案研究如何從 BCG（Ballistocardiogram，心沖擊圖）訊號估計心率，並分析訊號表示、諧波結構、感測退化與後處理對估計準確度的影響。

專案包含：

- HeartV6 心率估計與診斷工具；
- 控制式 amplitude-modulation 實驗；
- 真實 BCG 成功／失敗案例分析；
- 雜訊、振幅衰減、飽和與樣本遺失 robustness 實驗；
- non-learned reconstruction 實驗；
- 實驗設定、測試與研究論文。

研究論文原始檔位於 [`paper/main.tex`](paper/main.tex)，編譯版本位於 [`paper/main.pdf`](paper/main.pdf)。

## 授權與引用

程式碼採用 [MIT License](LICENSE)。論文、文件與原創圖表採用 [CC BY 4.0](LICENSE-CONTENT.md)。使用本專案進行研究時，請依照 [`CITATION.cff`](CITATION.cff) 引用來源。

本專案僅供研究使用，不作為醫療診斷工具。
