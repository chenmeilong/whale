import  pandas as  pd

csv_data = pd.read_csv("./data/test/label.csv",sep = ',')  # 读取验证数据
# print(csv_data.shape)
# print(csv_data.values)
a=csv_data.iloc[:,0].size#行数
# print(a)
# print(csv_data.columns)
label_dic={}
for i in csv_data.values:
    label_dic[i[0]]=i[1]
print(label_dic)

def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]
label_=get_key(label_dic,'w_17ee910')
print(label_[0])

submission_csv = pd.read_csv("./data/test/test_submission.csv",sep = ',')  # 读取验证数据
#print(submission_csv.iloc[:,1:3])
submission_data=submission_csv.iloc[:,1:3]  #行，列
print(len(submission_data))

for i in range(len(submission_data)):   #df.iloc[3:5,0:2]
    label_=get_key(label_dic,submission_data.iloc[i, 1])
    print(label_)
    submission_data.iloc[i, 1] = label_[0]
    #print(submission_data.iloc[i,1])
#print(submission_data)
submission_data.to_csv('./data/test/test.csv',index=False)              #保存到csv文件

