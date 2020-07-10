import predict
model = predict.load_model('./nsfw.299x299.h5')

#单个文件
print(predict.classify(model, './img/3.jpg'))
#多个文件
print(predict.classify(model, ['./img/1.png', './img/3.jpg']))
#文件夹
print(predict.classify(model, './img/'))
