
# Crowd Detection을 이용한 혼잡도 파악 후 안전사고 방지 프로그램

<img src="https://capsule-render.vercel.app/api?type=waving&color=BDBDC8&height=150&section=header" />

<img src="https://capsule-render.vercel.app/api?type=waving&color=BDBDC8&height=150&section=footer" />

## 개요

해당 프로젝트는 이태원 참사, 공연장 압사 사고등 단 시간에 많은 사람들이 몰려 인파를 제때 통제하지 못해 발생하는 사건사고들을 사전에 방지하기위한 프로젝트이다. 이를 구현하기위해 OpenVINO를 통한 객체인식을 활용하여 공간에 사람이
얼마나 몰려있는지를 공간 혼잡도를 시각적으로 파악한다. 결과적으로 임계값을 넘는 수준의 혼잡도가 발생하면 경고를 하거나 인파를 통제함으로써 발생할 수 있는 사고를 미연에 방지하는 기능을 제공한다.


## 프로젝트의 구성

프로젝트는 인파를 파악하는 CCTV화면에서 실시간으로 혼잡도를 파악하는 GUI를 제공한다.  


![video1](https://github.com/user-attachments/assets/7927c6a7-1f58-4b78-b057-8f2e1867607c)
