import os

import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 指定中文字體
plt.rcParams['font.family'] = ['Microsoft JhengHei']   # 或 ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號（避免被誤判為字元缺字）
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

from py_modules import common_utils

rootDir            = os.getcwd()                            # root directory
trainingHealthyDir = f'{rootDir}\\data\\LandingGear_relatives\\data\\training\\Healthy'  # training healthy data directory
trainingFaultyDir  = f'{rootDir}\\data\\LandingGear_relatives\\data\\training\\Faulty'   # training faulty data directory
testingDir         = f'{rootDir}\\data\\LandingGear_relatives\\data\\testing'            # testing data directory
csvDir             = f'{rootDir}\\data\\LandingGear_relatives\\csv'                      # csv dataset directory
featuresDir        = f'{rootDir}\\data\\LandingGear_relatives\\myfeature'                # features directory
modelDir           = f'{rootDir}\\data\\LandingGear_relatives\\model'                    # models directory

### Read features csv
healthyFeatures = pd.read_csv(f'{featuresDir}\\healthyAllFeatures.csv')
faultyFeatures   = pd.read_csv(f'{featuresDir}\\faultyAllFeatures.csv')
testingFeatures = pd.read_csv(f'{featuresDir}\\testingAllFeatures.csv')
'''
1. First we need to normalize the data features
    We can use all data including testing data, because we just want to fit the HI curve.
2. We need to do PCA to get our Principal Components.
3. Use PCs to build the Logistic Regression curve.
'''
# allFeatures     = pd.concat([healthyFeatures, faultyFeatures, testingFeatures], axis='rows', ignore_index=True)
allFeatures     = pd.concat([healthyFeatures, faultyFeatures], axis='rows', ignore_index=True)
featureName     = allFeatures.columns.tolist()

# pipe = Pipeline(steps=[
#     ('sc', StandardScaler()),
#     # ('pca', PCA(n_components=3, random_state=0)),
#     ('lr', LogisticRegression(random_state=0, solver='lbfgs'))
# ])
'''
由於PCA產出的特徵，在HI的極化表現不佳，所以改用自選特徵
'''
FEAT_IDXS = [0, 2, 3]  # 你要的三個欄位索引

# 1) 建立選欄器：只保留這三欄
sel = ColumnTransformer(
    transformers=[('pick', 'passthrough', FEAT_IDXS)],
    remainder='drop'
)

# 2) 建立完整 Pipeline：選欄 → 標準化 → LR
pipe = Pipeline(steps=[
    ('sel', sel),
    ('sc', StandardScaler()),
    ('lr', LogisticRegression(random_state=0, solver='lbfgs'))
])

### 1. Normalization (no need to save)
# sc = StandardScaler()
# allFeatures_norm = sc.fit_transform(allFeatures)
# joblib.dump(sc, f'{modelDir}\\sc_allfeatures1141017.pkl')  # save normalize model
# allFeatures_norm = pipe['sc'].fit_transform(allFeatures)


labels = [0]*healthyFeatures.shape[0] + [1]*faultyFeatures.shape[0]# + [1]*12 + [0]*9 + [1]*15 + [0]*16 + [1]*5 + [0]*8
# labels = [0]*healthyFeatures.shape[0] + [1]*faultyFeatures.shape[0] + [1]*12 + [0]*9 + [1]*15 + [0]*16 + [1]*5 + [0]*8  


pipe.fit(allFeatures, labels)
# 2️⃣ 取得「標準化後」的特徵（不會改變原資料）
X_sel = pipe.named_steps['sel'].transform(allFeatures)
X_norm = pipe.named_steps['sc'].transform(X_sel)

allFeatures_norm = allFeatures
print('================資料大小=================START', )
print('allFeatures_norm:', allFeatures.shape)
print('labels:', len(labels))

print('================資料大小=================END', )

# ### PCA
# # pca = PCA(n_components = n, random_state=0)         # Prepare PCA model
# # allFeatures_norm = pca.fit_transform(allFeatures_norm)  # Fit PCA model with all data
# # joblib.dump(pca, f'{modelDir}\\pca_onlyTrain1141019.pkl')  # save normalize model

# allFeatures_norm = pipe['pca'].fit_transform(allFeatures_norm)  # Fit PCA model with all data
# print('Shape of X:', allFeatures_norm.shape)


# ## PCA n components explained variance
# explained_variance = pipe['pca'].explained_variance_ratio_
# print(f'explained variance: {explained_variance}, total: {sum(explained_variance)}')

### Alternatives: Directly Select features
# no1_feature = 0
# no2_feature = 2
# no3_feature = 3
# no1_featureName = featureName[no1_feature]
# no2_featureName = featureName[no2_feature]
# no3_featureName = featureName[no3_feature]
# allFeatures_norm = allFeatures_norm[:, [no1_feature, no2_feature, no3_feature]]



### Train LR model
## Fitting Logistic Regression to the Training set
# LR_model = LogisticRegression(random_state = 0, solver='lbfgs') # Prepare LR model_0
# LR_model.fit(allFeatures_norm, labels) 
# pipe['lr'].fit(allFeatures_norm, labels)                                # Trainging LR model_0



iH = np.where(pipe.named_steps['lr'].classes_==0)[0][0]
HI_healthy = pipe.predict_proba(healthyFeatures.values)[:, iH]
HI_faulty  = pipe.predict_proba(faultyFeatures.values)[:, iH]
print('Healthy HI median:', np.median(HI_healthy))
print('Faulty  HI median:', np.median(HI_faulty))



## save model
# joblib.dump(LR_model, f'{modelDir}\\all_data_LR_Health_Indicator_Curve.pkl')
joblib.dump(pipe, f'{modelDir}\\hi_lr_pipeline_1141019_self_features.pkl')
print("Saved:", f'{modelDir}\\hi_lr_pipeline_1141019_self_features.pkl')

# === LR curve + scatter of each sample around the curve ===
# We use the trained LR_model on allFeatures_norm (n=3 PCs)

# 1) Linear scores (logits) and probabilities for all samples
# logit_all = LR_model.decision_function(allFeatures_norm)          # shape (N,)
logit_all = pipe.named_steps['lr'].decision_function(X_norm)
p_all     = 1.0 / (1.0 + np.exp(-logit_all))           # sigmoid(logit)

# 2) Smooth LR curve in the score domain
x_curve = np.linspace(logit_all.min() - 0.5, logit_all.max() + 0.5, 600)
y_curve = 1.0 / (1.0 + np.exp(-x_curve))               # sigmoid

# 3) Small jitter so points appear "around" the curve (purely for visualization)
rng = np.random.default_rng(0)
jitter = rng.normal(loc=0.0, scale=0.05, size=p_all.shape)   # ± ~1%
y_scatter = np.clip(p_all + jitter, 0.0, 1.0)

labels_np = np.array(labels)

plt.figure(figsize=(10, 6))

# LR curve
plt.plot(x_curve, y_curve, lw=3, label='LR curve (sigmoid of score)')

# Scatter points at their own score positions; color by class
plt.scatter(logit_all[labels_np == 0], y_scatter[labels_np == 0],
            s=45, c='#2E86AB', edgecolor='k', linewidth=0.6, alpha=0.85, label='Healthy (0)')
plt.scatter(logit_all[labels_np == 1], y_scatter[labels_np == 1],
            s=45, c='#E74C3C', edgecolor='k', linewidth=0.6, alpha=0.85, label='Faulty (1)')

# Cosmetics
plt.ylim(-0.02, 1.02)
plt.yticks(np.linspace(0, 1, 6))
# plt.xlabel('Linear score  s = w·x + b  (from LogisticRegression.decision_function)')
plt.xlabel('Logistic Regression 決策函數值')
# plt.ylabel('Predicted probability  P(Faulty)')
plt.ylabel('Logistic Regression 臨界故障預測值')
# plt.title('Logistic Regression Health Indicator\nLR curve with samples scattered around the curve')
plt.title('Logistic Regression Health Indicator')

plt.grid(True, alpha=0.3)
plt.legend(loc='best', frameon=True)

plt.tight_layout()
plt.savefig(f'{modelDir}\\LR_curve_with_samples.png', dpi=200)
plt.show()







