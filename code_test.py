def cv_morphology():
    code = """\
        import cv2
import numpy as np
from skimage.morphology import skeletonize, thin
import matplotlib.pyplot as plt

# --------------------------------------------------
# Padding Helper
# --------------------------------------------------
def pad_image(img, kernel):
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

# ==================================================
# ========  BINARY MORPHOLOGY IMPLEMENTATIONS  ======
# ==================================================
def erosion_binary(img, kernel):
    img_padded = pad_image(img, kernel)
    output = np.zeros_like(img)
    kh, kw = kernel.shape
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = img_padded[i:i+kh, j:j+kw]
            if np.all(region[kernel == 1] == 255):
                output[i, j] = 255
    return output

def dilation_binary(img, kernel):
    img_padded = pad_image(img, kernel)
    output = np.zeros_like(img)
    kh, kw = kernel.shape
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = img_padded[i:i+kh, j:j+kw]
            if np.any(region[kernel == 1] == 255):
                output[i, j] = 255
    return output

def opening_binary(img, kernel):
    return dilation_binary(erosion_binary(img, kernel), kernel)

def closing_binary(img, kernel):
    return erosion_binary(dilation_binary(img, kernel), kernel)

def gradient_binary(img, kernel):
    return dilation_binary(img, kernel) - erosion_binary(img, kernel)

def top_hat_white_binary(img, kernel):
    opened = opening_binary(img, kernel)
    return cv2.subtract(img, opened)

def top_hat_black_binary(img, kernel):
    closed = closing_binary(img, kernel)
    return cv2.subtract(closed, img)

def hit_or_miss(img, kernel):
    img_bin = (img // 255).astype(np.uint8)
    k1 = np.uint8(kernel == 1)
    k0 = np.uint8(kernel == 0)
    eroded1 = erosion_binary(img, k1)
    eroded0 = erosion_binary(255 - img, k0)
    return np.minimum(eroded1, eroded0)

def skeletonization(img):
    binary = (img // 255).astype(bool)
    skeleton = skeletonize(binary)
    return (skeleton * 255).astype(np.uint8)

def thinning(img):
    binary = (img // 255).astype(bool)
    thin_img = thin(binary)
    return (thin_img * 255).astype(np.uint8)

def pruning(skeleton_img):
    kernel = np.ones((3, 3), np.uint8)
    pruned = erosion_binary(skeleton_img, kernel)
    return pruned


# ==================================================
# ========  GRAY-LEVEL MORPHOLOGY IMPLEMENTS  ======
# ==================================================
def erosion_gray(img, kernel):
    img_padded = pad_image(img, kernel)
    output = np.zeros_like(img)
    kh, kw = kernel.shape
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = img_padded[i:i+kh, j:j+kw]
            output[i, j] = np.min(region[kernel == 1])
    return output

def dilation_gray(img, kernel):
    img_padded = pad_image(img, kernel)
    output = np.zeros_like(img)
    kh, kw = kernel.shape
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = img_padded[i:i+kh, j:j+kw]
            output[i, j] = np.max(region[kernel == 1])
    return output

def opening_gray(img, kernel):
    return dilation_gray(erosion_gray(img, kernel), kernel)

def closing_gray(img, kernel):
    return erosion_gray(dilation_gray(img, kernel), kernel)

def gradient_gray(img, kernel):
    return dilation_gray(img, kernel) - erosion_gray(img, kernel)

def top_hat_white_gray(img, kernel):
    opened = opening_gray(img, kernel)
    return cv2.subtract(img, opened)

def top_hat_black_gray(img, kernel):
    closed = closing_gray(img, kernel)
    return cv2.subtract(closed, img)


# ==================================================
# ================== MAIN EXECUTION =================
# ==================================================
if __name__ == "__main__":
    # Load grayscale image
    img = cv2.imread('cam.jpeg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found! Please check the path.")

    # Binary conversion
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Structuring element
    kernel = np.array([[0,1,0],
                       [1,1,1],
                       [0,1,0]], dtype=np.uint8)

    # ======== Binary Morphology ========
    ero_b = erosion_binary(binary, kernel)
    dil_b = dilation_binary(binary, kernel)
    opn_b = opening_binary(binary, kernel)
    cls_b = closing_binary(binary, kernel)
    grad_b = gradient_binary(binary, kernel)
    topw_b = top_hat_white_binary(binary, kernel)
    topb_b = top_hat_black_binary(binary, kernel)
    hm = hit_or_miss(binary, kernel)
    skel = skeletonization(binary)
    thin_img = thinning(binary)
    pruned = pruning(skel)

    # ======== Gray-Level Morphology ========
    ero_g = erosion_gray(img, kernel)
    dil_g = dilation_gray(img, kernel)
    opn_g = opening_gray(img, kernel)
    cls_g = closing_gray(img, kernel)
    grad_g = gradient_gray(img, kernel)
    topw_g = top_hat_white_gray(img, kernel)
    topb_g = top_hat_black_gray(img, kernel)

    # ======== Display All ========
    titles = [
        'Binary Erosion', 'Binary Dilation', 'Binary Opening', 'Binary Closing',
        'Binary Gradient', 'Top Hat (White)', 'Top Hat (Black)', 'Hit-or-Miss',
        'Skeletonization', 'Thinning', 'Pruning',
        'Gray Erosion', 'Gray Dilation', 'Gray Opening', 'Gray Closing',
        'Gray Gradient', 'Gray Top Hat (White)', 'Gray Top Hat (Black)'
    ]
    images = [
        ero_b, dil_b, opn_b, cls_b,
        grad_b, topw_b, topb_b, hm,
        skel, thin_img, pruned,
        ero_g, dil_g, opn_g, cls_g,
        grad_g, topw_g, topb_g
    ]

    plt.figure(figsize=(18, 18))
    for i in range(len(images)):
        plt.subplot(5, 4, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
"""
    return code 

def cv_feature():
    code = """\
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import blob_log
import importlib

# --- Handle scikit-image GLCM imports for both old and new versions ---
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix
    from skimage.feature import greycoprops as graycoprops

# ========== 1. Read Image in Grayscale ==========
img = cv2.imread('cam.jpeg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis('off')
plt.show()

# ========== 2. Corner Detection ==========
# Harris Corner Detection
harris = cv2.cornerHarris(np.float32(img), 2, 3, 0.04)
harris = cv2.dilate(harris, None)
img_harris = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_harris[harris > 0.01 * harris.max()] = [0, 0, 255]

# Shi-Tomasi Corner Detection
corners = cv2.goodFeaturesToTrack(img, 100, 0.01, 10)
img_shi = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
if corners is not None:
    for c in corners:
        x, y = c.ravel()
        cv2.circle(img_shi, (int(x), int(y)), 3, (0, 255, 0), -1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_harris, cv2.COLOR_BGR2RGB))
plt.title("Harris Corners")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_shi, cv2.COLOR_BGR2RGB))
plt.title("Shi-Tomasi Corners")
plt.axis('off')
plt.show()

# ========== 3. Blob Detection (Laplacian of Gaussian) ==========
blobs = blob_log(img, max_sigma=30, num_sigma=10, threshold=0.05)
img_blob = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for blob in blobs:
    y, x, r = blob
    cv2.circle(img_blob, (int(x), int(y)), int(r * np.sqrt(2)), (255, 0, 0), 2)

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img_blob, cv2.COLOR_BGR2RGB))
plt.title("Blob Detection (LoG)")
plt.axis('off')
plt.show()

# ========== 4. Texture Analysis (GLCM) ==========
glcm = graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                    levels=256, symmetric=True, normed=True)

contrast = graycoprops(glcm, 'contrast')
homogeneity = graycoprops(glcm, 'homogeneity')
energy = graycoprops(glcm, 'energy')
correlation = graycoprops(glcm, 'correlation')

print("=== Texture Analysis (GLCM Features) ===")
print(f"Contrast: {contrast.mean():.4f}")
print(f"Homogeneity: {homogeneity.mean():.4f}")
print(f"Energy: {energy.mean():.4f}")
print(f"Correlation: {correlation.mean():.4f}")

# ========== 5. Feature Descriptors (SIFT, SURF, ORB) ==========

# --- SIFT ---
sift = cv2.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(img, None)
img_sift = cv2.drawKeypoints(img, kp_sift, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



# --- ORB ---
orb = cv2.ORB_create()
kp_orb, des_orb = orb.detectAndCompute(img, None)
img_orb = cv2.drawKeypoints(img, kp_orb, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# --- Display SIFT, SURF, ORB ---
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
plt.title("SIFT Features")
plt.axis('off')



plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
plt.title("ORB Features")
plt.axis('off')
plt.show()

"""
    return code 


def dl_rnn_csv():
    code = """\
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load CSV
df = pd.read_csv('data.csv')
values = df['value'].values.astype(float)

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data)-seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs), np.array(ys)

SEQ_LEN = 5
X, y = create_sequences(values, SEQ_LEN)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_losses, val_losses = [], []
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(X_train)
    loss = criterion(out.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred.squeeze(), y_val)
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

# Train vs Val loss
plt.figure(figsize=(8,4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('RNN Train vs Validation Loss')
plt.legend()
plt.show()

# Predicted vs Actual
with torch.no_grad():
    preds = model(X_val).squeeze().numpy()
plt.figure(figsize=(8,4))
plt.plot(y_val, label='Actual')
plt.plot(preds, label='Predicted')
plt.title('RNN Predictions vs Actual')
plt.legend()
plt.show()

#  Residuals and Histogram
residuals = y_val.numpy() - preds
plt.figure(figsize=(8,4))
plt.plot(residuals, color='r')
plt.axhline(0, color='black', linestyle='--')
plt.title('Prediction Residuals')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=30, kde=True, color='teal')
plt.title('Residual Error Distribution')
plt.show()

# Correlation heatmap of input sequences
corr = pd.DataFrame(X_val.squeeze().numpy()).corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Input Sequence Correlation')
plt.show()

# Confusion matrix (convert regression to discrete bins)
bins = np.linspace(min(values), max(values), 5)
y_true_binned = np.digitize(y_val, bins)
y_pred_binned = np.digitize(preds, bins)
cm = confusion_matrix(y_true_binned, y_pred_binned)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Purples')
plt.title('Confusion Matrix (Discretized Regression)')
plt.show()


"""

    return code


def dl_autoencoder():
    code = """\
        import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Load data
df = pd.read_csv('data.csv')
X = torch.tensor(df.values, dtype=torch.float32)

class AE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 4))
        self.decoder = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, input_dim))
    def forward(self, x): return self.decoder(self.encoder(x))
    def encode(self, x): return self.encoder(x)

model = AE(X.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, X)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Loss curve
plt.plot(losses, label='Training Loss')
plt.title('Autoencoder Loss Curve')
plt.legend()
plt.show()

# Reconstruction comparison
with torch.no_grad():
    rec = model(X).numpy()
err = X.numpy() - rec

plt.plot(X[:,0], label='Original')
plt.plot(rec[:,0], label='Reconstructed')
plt.legend(); plt.title('Original vs Reconstructed (Feature 1)')
plt.show()

# Error heatmap
sns.heatmap(err, cmap='coolwarm', center=0)
plt.title('Reconstruction Error Heatmap')
plt.show()

# Error histogram
sns.histplot(err.flatten(), bins=50, kde=True)
plt.title('Reconstruction Error Distribution')
plt.show()



# Confusion matrix (error region classification)
bins = np.linspace(err.min(), err.max(), 5)
true_binned = np.digitize(X[:,0], bins)
pred_binned = np.digitize(rec[:,0], bins)
cm = confusion_matrix(true_binned, pred_binned)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Reds')
plt.title('Confusion Matrix (Binned Reconstruction)')
plt.show()

"""
    return code

def dl_rnn_text():
    code = """\
        import torch, torch.nn as nn, seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Load and encode text
text = open('text.txt','r',encoding='utf-8').read()
chars = sorted(list(set(text)))
c2i = {c:i for i,c in enumerate(chars)}
i2c = {i:c for i,c in enumerate(chars)}
enc = [c2i[c] for c in text]
SEQ_LEN = 40
X, y = [], []
for i in range(len(enc)-SEQ_LEN):
    X.append(enc[i:i+SEQ_LEN])
    y.append(enc[i+SEQ_LEN])
X, y = torch.tensor(X), torch.tensor(y)

class CharRNN(nn.Module):
    def __init__(self, vocab, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.rnn = nn.RNN(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)
    def forward(self, x):
        e = self.embed(x)
        out,_ = self.rnn(e)
        return self.fc(out[:,-1,:])

vocab = len(chars)
model = CharRNN(vocab)
opt = torch.optim.Adam(model.parameters(), lr=0.003)
crit = nn.CrossEntropyLoss()
losses, accs = [], []

for epoch in range(30):
    opt.zero_grad()
    out = model(X)
    loss = crit(out, y)
    losses.append(loss.item())
    preds = torch.argmax(out, dim=1)
    acc = (preds == y).float().mean().item()
    accs.append(acc)
    loss.backward()
    opt.step()
    if (epoch+1)%5==0:
        print(f'Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.2f}')

# Loss & Accuracy
fig,ax=plt.subplots(1,2,figsize=(10,4))
ax[0].plot(losses); ax[0].set_title('Loss')
ax[1].plot(accs); ax[1].set_title('Accuracy')
plt.show()

# Confusion matrix (chars)
cm = confusion_matrix(y.numpy(), preds.numpy(), labels=list(range(vocab)))
ConfusionMatrixDisplay(cm, display_labels=chars).plot(cmap='Blues')
plt.title('Character Confusion Matrix')
plt.xticks(rotation=90)
plt.show()

# Probability heatmap
probs = torch.softmax(out, dim=1).detach().numpy()
sns.heatmap(probs[:50], cmap='YlGnBu')
plt.title('Character Probability Heatmap (First 50 Samples)')
plt.xlabel('Character Index')
plt.ylabel('Sample Index')
plt.show()

"""
    return code 
