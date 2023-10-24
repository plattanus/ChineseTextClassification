# import matplotlib.pyplot as plt
#
# # epoch,acc,loss,val_acc,val_loss
# x_axis_data = ['TextCNN', 'DPCNN', 'TextRNN', 'TextRNN+Att', 'TextRCNN', 'FastText']
# y_axis_data = [90.73, 90.88, 90.41, 90.48, 91.01, 91.58]
#
# # 画图
# plt.plot(x_axis_data, y_axis_data, 'b*--', alpha=0.5, linewidth=1, label='acc')  # '
#
# ## 设置数据标签位置及大小
# for a, b in zip(x_axis_data, y_axis_data):
#     plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)  # ha='center', va='top'
# plt.legend()  # 显示上面的label
#
# plt.xlabel('Model')
# plt.ylabel('Accuracy')  # accuracy
#
# plt.show()




import matplotlib.pyplot as plt

# epoch,acc,loss,val_acc,val_loss
x_axis_data = ['Finance','Realty','Stocks','Education','Science',
               'Society','Politics','Sports','Game','Entertainment']
y_Precision_data_TextCNN = [0.9050,0.9011,0.8390,0.9463,0.8591,
                       0.8885,0.9268,0.9607,0.9493,0.9356]
y_Precision_data_DPCNN = [0.8892,0.9020,0.8870,0.9330,0.8470,
                          0.8989,0.9040,0.9828,0.9301,0.9133]
y_Precision_data_TextRNN = [0.8979,0.9029,0.8920,0.9391,0.8231,
                            0.8683,0.8905,0.9691,0.9643,0.9043]
y_Precision_data_TextRNNAtt = [0.9163,0.9035,0.8571,0.9536,0.8142,
                               0.9068,0.8849,0.9588,0.9653,0.8988]
y_Precision_data_TextRCNN = [0.9040,0.9411,0.8666,0.9601,0.8130,
                             0.9090,0.8811,0.9730,0.9550,0.9108]
y_Precision_data_FastText = [0.9428,0.9218,0.8477,0.9355,0.8904,
                             0.9094,0.8836,0.9807,0.9517,0.9268]

# # 画图
# plt.plot(x_axis_data, y_Precision_data_TextCNN, 'b*--', alpha=0.5, linewidth=1, label='TextCNN')  # '
# plt.plot(x_axis_data, y_Precision_data_DPCNN, 'rs--', alpha=0.5, linewidth=1, label='DPCNN')
# plt.plot(x_axis_data, y_Precision_data_TextRNN, 'go--', alpha=0.5, linewidth=1, label='TextRNN')
# plt.plot(x_axis_data, y_Precision_data_TextRNNAtt, 'kp--', alpha=0.5, linewidth=1, label='TextRNNAtt')  # '
# plt.plot(x_axis_data, y_Precision_data_TextRCNN, 'y+--', alpha=0.5, linewidth=1, label='TextRCNN')
plt.plot(x_axis_data, y_Precision_data_FastText, 'cv--', alpha=0.5, linewidth=1, label='FastText')
#
# ## 设置数据标签位置及大小
# for a, b in zip(x_axis_data, y_Precision_data_TextCNN):
#     plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)  # ha='center', va='top'
# for a, b1 in zip(x_axis_data, y_Precision_data_DPCNN):
#     plt.text(a, b1, str(b1), ha='center', va='bottom', fontsize=8)
# for a, b2 in zip(x_axis_data, y_Precision_data_TextRNN):
#     plt.text(a, b2, str(b2), ha='center', va='bottom', fontsize=8)
# for a, b3 in zip(x_axis_data, y_Precision_data_TextRNNAtt):
#     plt.text(a, b3, str(b3), ha='center', va='bottom', fontsize=8)  # ha='center', va='top'
# for a, b4 in zip(x_axis_data, y_Precision_data_TextRCNN):
#     plt.text(a, b4, str(b4), ha='center', va='bottom', fontsize=8)
for a, b5 in zip(x_axis_data, y_Precision_data_FastText):
    plt.text(a, b5, str(b5), ha='center', va='bottom', fontsize=8)

# ==============================================================================================

y_Recall_data_TextCNN = [0.8760, 0.9480, 0.8700, 0.9510, 0.8780,
                         0.9240, 0.8730, 0.9540, 0.8990, 0.9300]
y_Recall_data_DPCNN = [0.8910, 0.9390, 0.7850, 0.9610, 0.8690,
                       0.9160, 0.8760, 0.9710, 0.9320, 0.9480]
y_Recall_data_TextRNN = [0.8970, 0.9300, 0.8090, 0.9400, 0.8750,
                         0.9360, 0.8540, 0.9730, 0.8910, 0.9360]
y_Recall_data_TextRNNAtt = [0.8870, 0.9360, 0.8280, 0.9240, 0.8810,
                            0.9050, 0.8690, 0.9770, 0.8910, 0.9500]
y_Recall_data_TextRCNN = [0.8950, 0.9100, 0.8250, 0.9380, 0.9000,
                          0.9090, 0.8890, 0.9730, 0.9120, 0.9500]
y_Recall_data_FastText = [0.8740, 0.9430, 0.8850, 0.9580, 0.8690,
                          0.9030, 0.9030, 0.9680, 0.9450, 0.9370]

# # 画图
# plt.plot(x_axis_data, y_Recall_data_TextCNN, 'b*--', alpha=0.5, linewidth=1, label='TextCNN')  # '
# plt.plot(x_axis_data, y_Recall_data_DPCNN, 'rs--', alpha=0.5, linewidth=1, label='DPCNN')
# plt.plot(x_axis_data, y_Recall_data_TextRNN, 'go--', alpha=0.5, linewidth=1, label='TextRNN')
# plt.plot(x_axis_data, y_Recall_data_TextRNNAtt, 'kp--', alpha=0.5, linewidth=1, label='TextRNNAtt')  # '
# plt.plot(x_axis_data, y_Recall_data_TextRCNN, 'y+--', alpha=0.5, linewidth=1, label='TextRCNN')
plt.plot(x_axis_data, y_Recall_data_FastText, 'cv--', alpha=0.5, linewidth=1, label='FastText')
#
# ## 设置数据标签位置及大小
# for a, b in zip(x_axis_data, y_Recall_data_TextCNN):
#     plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)  # ha='center', va='top'
# for a, b1 in zip(x_axis_data, y_Recall_data_DPCNN):
#     plt.text(a, b1, str(b1), ha='center', va='bottom', fontsize=8)
# for a, b2 in zip(x_axis_data, y_Recall_data_TextRNN):
#     plt.text(a, b2, str(b2), ha='center', va='bottom', fontsize=8)
# for a, b3 in zip(x_axis_data, y_Recall_data_TextRNNAtt):
#     plt.text(a, b3, str(b3), ha='center', va='bottom', fontsize=8)  # ha='center', va='top'
# for a, b4 in zip(x_axis_data, y_Recall_data_TextRCNN):
#     plt.text(a, b4, str(b4), ha='center', va='bottom', fontsize=8)
for a, b6 in zip(x_axis_data, y_Recall_data_FastText):
    plt.text(a, b6, str(b6), ha='center', va='bottom', fontsize=8)

# ==============================================================================================

y_F1score_data_TextCNN = [0.8902, 0.9240, 0.8542, 0.9486, 0.8684,
                          0.9059, 0.8991, 0.9574, 0.9235, 0.9328]
y_F1score_data_DPCNN = [0.8901, 0.9201, 0.8329, 0.9468, 0.8578,
                        0.9074, 0.8898, 0.9769, 0.9311, 0.9303]
y_F1score_data_TextRNN = [0.8974, 0.9163, 0.8485, 0.9395, 0.8483,
                          0.9009, 0.8719, 0.9711, 0.9262, 0.9199]
y_F1score_data_TextRNNAtt = [0.9014, 0.9194, 0.8423, 0.9385, 0.8463,
                             0.9059, 0.8769, 0.9678, 0.9267, 0.9237]
y_F1score_data_TextRCNN = [0.8995, 0.9253, 0.8453, 0.9489, 0.8543,
                           0.9090, 0.8850, 0.9730, 0.9330, 0.9300]
y_F1score_data_FastText = [0.9071, 0.9323, 0.8659, 0.9466, 0.8796,
                           0.9062, 0.8932, 0.9743, 0.9483, 0.9319]
#
# # 画图
# plt.plot(x_axis_data, y_F1score_data_TextCNN, 'b*--', alpha=0.5, linewidth=1, label='TextCNN')  # '
# plt.plot(x_axis_data, y_F1score_data_DPCNN, 'rs--', alpha=0.5, linewidth=1, label='DPCNN')
# plt.plot(x_axis_data, y_F1score_data_TextRNN, 'go--', alpha=0.5, linewidth=1, label='TextRNN')
# plt.plot(x_axis_data, y_F1score_data_TextRNNAtt, 'kp--', alpha=0.5, linewidth=1, label='TextRNNAtt')  # '
# plt.plot(x_axis_data, y_F1score_data_TextRCNN, 'y+--', alpha=0.5, linewidth=1, label='TextRCNN')
plt.plot(x_axis_data, y_F1score_data_FastText, 'cv--', alpha=0.5, linewidth=1, label='FastText')
#
# ## 设置数据标签位置及大小
# for a, b in zip(x_axis_data, y_F1score_data_TextCNN):
#     plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)  # ha='center', va='top'
# for a, b1 in zip(x_axis_data, y_F1score_data_DPCNN):
#     plt.text(a, b1, str(b1), ha='center', va='bottom', fontsize=8)
# for a, b2 in zip(x_axis_data, y_F1score_data_TextRNN):
#     plt.text(a, b2, str(b2), ha='center', va='bottom', fontsize=8)
# for a, b3 in zip(x_axis_data, y_F1score_data_TextRNNAtt):
#     plt.text(a, b3, str(b3), ha='center', va='bottom', fontsize=8)  # ha='center', va='top'
# for a, b4 in zip(x_axis_data, y_F1score_data_TextRCNN):
#     plt.text(a, b4, str(b4), ha='center', va='bottom', fontsize=8)
for a, b7 in zip(x_axis_data, y_F1score_data_FastText):
    plt.text(a, b7, str(b7), ha='center', va='bottom', fontsize=8)

# ==============================================================================================

plt.legend()  # 显示上面的label

plt.xlabel('category')
plt.ylabel('efficiency')

# plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()



