---
name: 問題回報
about: 執行時出錯、行為不如預期
labels: bug
---

## 發生什麼事

（簡述）

## 重現方式

```
你執行的完整指令
```

## 執行環境回報 ← 最重要

請執行下列指令並把產生的 `report.json` 內容貼上（或附檔）：

```
focus-keeper.exe --source 0 --no-display --duration 10 --report report.json
```

原始碼版：

```
python -m focus_keeper.cli --source 0 --no-display --duration 10 --report report.json
```

<details>
<summary>report.json</summary>

```json
貼在這裡
```

</details>

裡面有套件版本、模型 SHA-256、設備資訊與效能數據，沒有這個很難判斷問題。

## 終端輸出

```
把 [error] / [warn] 開頭的訊息貼上
```

## 先確認過了嗎

- [ ] 讀過 [INSTALL.md 的疑難排解](../../blob/main/INSTALL.md#疑難排解)
- [ ] 讀過 [README 的能力界線](../../blob/main/README.md#能力界線使用前必讀)（例如：`PRIMARY_LEFT` 不等於人離開了）
