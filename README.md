# FaceRecognition
실시간 얼굴인식 API개발
실시간 영상으로부터 얼굴 특징점 60개를 검출, 눈,코,입,눈썹,얼굴선에 대한 각각의 특징점들을 1차원 배열의 형태로 반환함.
- main.py : 1명의 얼굴만 인식 가능
- Test.py : 여러명의 얼굴 인식, 사진을 바탕으로 누구인지 확인가능. 1프레임당 처리시간 평균 0.4초로 실시간 사용은 어려움
- 앞면 얼굴 인식

## 사용 모듈
- OpenCV with python3
- face_recognition module
- numpy
- dlib

## Description
![1](./1.png)
