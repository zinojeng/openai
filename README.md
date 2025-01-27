# Translation Agent

Translation Agent 是一個基於大型語言模型的翻譯工具，採用獨特的「省思式工作流程」來提升翻譯品質。它模擬人類翻譯專家的思考過程，通過多步驟的翻譯和優化來產生高品質的譯文。

## 特點

- **多模型支援**：
  - OpenAI 模型：GPT-4, GPT-4-mini, GPT-3.5 Turbo
  - DeepSeek 模型：DeepSeek-Chat, DeepSeek-Reasoner

- **多語言支援**：
  - 源語言：中文、英文、西班牙文、法文、德文、義大利文、日文、韓文、越南文、印尼文、泰文
  - 目標語言：英文、繁體中文、簡體中文、西班牙文、法文、德文、義大利文、日文、韓文、越南文、印尼文、泰文

- **多種翻譯模式**：
  - Standard：標準翻譯
  - Fluency：流暢優先
  - Natural：自然口語化
  - Formal：正式文件
  - Academic：學術論文
  - Simple：簡單易懂
  - Creative：創意翻譯
  - Expand：擴充解釋
  - Shorten：精簡摘要

- **多種輸入格式**：
  - 直接文字輸入
  - PDF 文件
  - TXT 文件
  - Word 文件 (DOCX)

- **醫療專科支援**：
  - 內科系統
  - 外科系統
  - 其他專科

## 工作流程

1. **初始翻譯**：使用選定的語言模型進行初步翻譯
2. **反思與改進**：AI 分析初步譯文，提出改進建議
3. **優化輸出**：根據反思結果優化譯文，提升準確度和流暢度

## 成本估算

- 即時顯示 token 使用量
- 自動計算翻譯成本（新台幣）
- 支援不同模型的價格設定

## 使用方法

1. 在側邊欄選擇翻譯模型
2. 輸入相應的 API 金鑰（OpenAI 或 DeepSeek）
3. 選擇源語言和目標語言
4. 選擇翻譯模式
5. 輸入要翻譯的文字或上傳文件
6. 點擊「Translate」開始翻譯

## 輸出結果

- 初始翻譯結果
- 翻譯反思與建議
- 文本差異比較
- 最終優化譯文
- Token 使用統計和成本估算

## 開發者

- **修改者**：Tseng Yao Hsien
- **聯絡方式**：zinojeng@gmail.com
- **參考來源**：基於 Andrew Ng 的 AI Agent System for Language Translation

## 授權

本專案採用 MIT 授權條款。

## 系統需求

- Python 3.8+
- Streamlit
- OpenAI API 或 DeepSeek API 金鑰
- 相關 Python 套件（requirements.txt）

## 安裝

### 1. 複製專案
```bash
git clone https://github.com/zinojeng/openai.git
cd openai
```

### 2. 安裝依賴套件
```bash
pip install -r requirements.txt
```

### 3. 啟動應用程式
```bash
streamlit run paired_app.py
```

### 必要套件
```text
streamlit
openai
tiktoken
nltk
PyPDF2
python-docx
diff-match-patch
httpx
```

## 注意事項

- API 金鑰請妥善保管，不要上傳至公開場所
- 翻譯成本依據選擇的模型和文本長度而定
- 建議在翻譯較長文本時使用較經濟的模型

## 更新日誌

### 2024-03-21
- 新增 DeepSeek 模型支援
  - DeepSeek-Chat：適合一般翻譯任務
  - DeepSeek-Reasoner：適合需要深度推理的翻譯
- 優化 API 金鑰管理
  - 根據選擇的模型動態顯示對應的 API 金鑰輸入欄位
  - 改進錯誤處理和提示訊息

### 2024-03-20
- 改進成本估算系統
  - 新增各模型的精確定價
  - 支援 NTD 即時換算
- 優化使用者介面
  - 新增文本差異比較功能
  - 改進翻譯結果展示方式

### 2024-03-19
- 新增多種翻譯模式
  - 支援 9 種不同的翻譯風格
  - 可調整溫度參數
- 加入醫療專科支援
  - 新增完整的醫療科別選項
  - 支援專業術語翻譯

### 2024-03-18
- 初始版本發布
  - 基本翻譯功能
  - 多語言支援
  - 文件上傳功能

## 未來規劃

- [ ] 批量翻譯功能
- [ ] 術語表管理系統
- [ ] 翻譯記憶庫
- [ ] API 使用量監控
- [ ] 自動備份功能
- [ ] 多人協作功能

## 貢獻指南

歡迎提交 Pull Request 或開設 Issue 來改進專案。在提交之前，請確保：

1. 程式碼遵循 PEP 8 規範
2. 新功能包含適當的測試
3. 更新文件以反映變更
4. 在更新日誌中記錄重要改動

## 問題回報

如果您發現任何問題或有改進建議，請：

1. 檢查是否已有相關的 Issue
2. 提供詳細的問題描述
3. 附上重現問題的步驟
4. 如果可能，提供錯誤日誌或截圖

## 部署說明

### Streamlit Cloud 部署注意事項

1. **應用程式休眠**
   - Streamlit Cloud 免費版本在一段時間無人訪問後會自動休眠
   - 首次訪問時可能需要等待 30-60 秒喚醒

2. **保持活躍的方法**
   - 使用定期訪問服務（如 UptimeRobot）
   - 設置應用程式自我喚醒機制
   - 考慮升級到 Streamlit Cloud Teams 版本

3. **監控建議**
   - 使用 UptimeRobot 設置監控（免費版本）
   - 監控間隔：建議設置為 5 分鐘
   - 添加通知以便及時了解應用狀態

4. **最佳實踐**
   - 重要場合建議提前訪問喚醒應用
   - 可以使用瀏覽器擴展定期訪問
   - 考慮使用付費版本避免休眠問題