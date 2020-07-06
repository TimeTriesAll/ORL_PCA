import numpy as np
import  matplotlib.image as mpimg
from scipy import misc

# 图像缩放比例
scale=0.5
# 图像五五分
k=5
# 改变图像大小
def img2vector(filename):
    imgVector = misc.imresize(mpimg.imread(filename), scale).flatten()
    return imgVector.astype(np.float)
# 加载数据
def loadimage(dataSetDir):
    train_face = np.zeros((40 * k, int(112 * scale) * int(92 * scale)))  # 图像尺寸:112*92
    train_face_number = np.zeros(40 * k).astype(np.int8)
    test_face = np.zeros((40 * (10 - k), int(112 * scale) * int(92 * scale)))
    test_face_number = np.zeros(40 * (10 - k)).astype(np.int8)
    for i in np.linspace(1, 40, 40).astype(np.int8): #40个人
            people_num = i
            for j in np.linspace(1, 10, 10).astype(np.int8): #每个人有十张不同的脸
                if j <= k:
                    filename = dataSetDir+'/s'+str(people_num)+'/'+str(j)+'.pgm'
                    img = img2vector(filename)
                    train_face[(i-1)*k+(j-1),:] = img
                    train_face_number[(i-1)*k+(j-1)] = people_num
                else:
                    filename = dataSetDir+'/s'+str(people_num)+'/'+str(j)+'.pgm'
                    img = img2vector(filename)
                    test_face[(i-1)*(10-k)+(j-k)-1,:] = img
                    test_face_number[(i-1)*(10-k)+(j-k)-1] = people_num
    return train_face,train_face_number,test_face,test_face_number
# 原始数据（训练数据，测试数据）均减去训练数据均值
def subvector(target_matrix, target_vector):
    vector4matrix = np.repeat(target_vector, target_matrix.shape[0],axis = 0)
    target_matrix = target_matrix - vector4matrix
    return target_matrix
# 求均值
def submean(train_data, test_data):
    mean_data = train_data.mean(axis = 0).reshape(1, train_data.shape[1])
    train_data = subvector(train_data, mean_data)
    test_data = subvector(test_data, mean_data)
    return train_data,test_data
# 主函数
def main():
    # 加载数据
    train_face,train_face_number,test_face,test_face_number = loadimage('C:\\Users\\TimeT\\Desktop\\ORL_Faces')
    # 求均值
    train_face, test_face = submean(train_face, test_face)
    # 计算协方差，进行降维
    cov = np.dot(train_face.T, train_face)
    # 计算特征值和特征向量
    l, v = np.linalg.eig(cov)
    # 按特征值由大到小对特征向量排序
    mix = np.vstack((l,v))  # 按垂直方向（行顺序）堆叠数组构成一个新的数组
    mix = mix.T[np.lexsort(mix[::-1,:])].T[:,::-1]  # 排序
    v = np.delete(mix, 0, axis = 0) # 删除相关性小的特征向量，选择较大特征值对应的特征向量
    # 选择主要成分并将人脸图像映射到高维空间
    v = v[:,0:int(v.shape[1])]
    train_face = np.dot(train_face, v)
    test_face = np.dot(test_face , v)
    # 通过测量高维空间中的欧几里得距离来识别
    count = 0
    for i in np.linspace(0, test_face.shape[0] - 1, test_face.shape[0]).astype(np.int64):
        sub = subvector(train_face,test_face[i, :].reshape((1,test_face.shape[1])))
        dis = np.linalg.norm(sub, axis = 1)
        fig = np.argmin(dis)    #表示最小值在数组中所在的位置
        if train_face_number[fig] == test_face_number[i]:
            count = count + 1
    correct_rate = count / test_face.shape[0]
    print('特征向量个数：{}个'.format(int(v.shape[1])))
    print('准确率 = {} %'.format(correct_rate * 100))

main()
