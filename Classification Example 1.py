import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

# مسار ملف البيانات
path = 'F:\\3.txt'

# قراءة البيانات باستخدام مكتبة Pandas
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

# طباعة البيانات للاطلاع عليها
print('data = ')
print(data.head(10))
print()
print('data.describe = ')
print(data.describe())

# فصل الأمثلة الإيجابية والسلبية لرسمها على الرسم البياني
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')

# تعريف وظيفة التشبع (Sigmoid)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# رسم الدالة التشبع
nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(nums, sigmoid(nums), 'r')

# تعريف وظيفة 
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

# إضافة عمود من القيم الواحدة إلى البيانات (لتسهيل عملية الضرب المصفوفي)
data.insert(0, 'Ones', 1)

# تحديد مصفوفة X (البيانات) و y (النتائج المستهدفة)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

# تحويل البيانات إلى مصفوفات NumPy وتهيئة مصفوفة المعلمات theta
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.zeros(3)

print()
print('X.shape = ', X.shape)
print('theta.shape = ', theta.shape)
print('y.shape = ', y.shape)

# حساب تكلفة النموذج قبل عملية الأمثلية
thiscost = cost(theta, X, y)
print()
print('cost = ', thiscost)

# تعريف وظيفة حساب التدرج (الجزئي)
def gradient(theta, X, y):
    theta = np.matrix(theta)
   
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)
    return grad

# الأمثلية باستخدام تقنية تدريج النقص الثابت
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
costafteroptimize = cost(result[0], X, y)

print()
print('cost after optimize = ', costafteroptimize)
print()

# تعريف وظيفة التنبؤ
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

# تحديد مصفوفة الثيتا بعد عملية الأمثلية
theta_min = np.matrix(result[0])

# التنبؤ باستخدام النموذج المحسن
predictions = predict(theta_min, X)
print(predict(theta_min, X))
# حساب دقة النموذج
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0))
           else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))
