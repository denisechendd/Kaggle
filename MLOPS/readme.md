***This folder contains several Python scripts that demonstrate various aspects of Machine Learning Operations (MLOPS) such as data preprocessing, model training, and deployment.*** 

1. DashApp_example.py
Purpose: 用於建立一個 Dash 應用程式，進行房地產價格預測。在此應用中，使用者輸入房屋特徵（如距離捷運的距離、便利商店數量等），應用會基於訓練好的線性回歸模型預測單位面積房價。
Key Features:
載入一個房價資料用於訓練線性迴歸模型
建立一個互動式網頁介面使用Dash，讓使用者可以看到每個MRT站點距離作為模型特徵值，MRT站點的距離向量作為預測房價特徵
這可以參考介面照片：Real Estate APP screen.png

3. PreProcess_DAG.py
Purpose: 使用 Apache Airflow 建立資料預處理管道。該腳本負責清理、轉換和特徵工程，並將處理好的資料存儲為新的 CSV 檔案。
Key Features:
作為前處理APP用量資料，並操作特徵值的工程處理像是將數值資料取LOG和將類別型資料做特徵工程Encoding
定義一個Airflow DAG 和自動化前處理工作讓它每日運行

5. resnet.py
Purpose: 定義 ResNet 深度學習模型的相關架構，包括 ResNet-18、ResNet-34 等。可用於圖像分類或其他深度學習任務（例如：基於 PyTorch）。
Key Features:
新增一個ResNet架構的模型例如ResNet-18, ResNet-50, and ResNet-101.
作為新增一個Residual network 特徵的模型並使用(BasicBlock and Bottleneck)的模塊開發

7. train.py
Purpose: 用於訓練基於 BPR（Bayesian Personalized Ranking）的推薦系統模型。包括數據加載、模型訓練、損失計算和模型評估。
Key Features:
實行資料準備和處理計算遺失值（loss function）用於記分類的模型
支持計算評估矩陣如precision and recall於各個的恆量值
Allows for periodic saving of the trained model and logging of training metrics.

9. transform.py
Purpose: 定義 GeneralizedRCNNTransform 類別，用於對影像數據進行預處理，包括影像歸一化、重新調整大小、批處理等，主要用於目標檢測模型（如 Faster-RCNN）的數據管道。

11. Case_Study_PCA_and_TSNE_MLS.ipynb：
Purpose: 包含主題為 PCA（主成分分析）及 t-SNE（t分佈鄰域嵌入）的機器學習案例研究。


