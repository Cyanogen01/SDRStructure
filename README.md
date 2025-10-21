# SDRStructure
__SDRStructure__: A Scalable Deep Recurrent Structure for Battery Health Management

## Resources
### [__Paper__](https://ieeexplore.ieee.org/document/11017404) | [__Wechat Article__](https://mp.weixin.qq.com/s/FXt4EfbD-2cqx5jX3kKEzw) | [__DIB-Lab Webcite__](https://dib-lab.com/)

## Abstract
Accurately evaluating battery state-of-health and lifetime prognostics with uncertainty quantification is important to guarantee the reliability, safety, and efficiency of batteries’ deployment. The Gaussian process-based data-driven solutions have become one of the most widely used methods due to their non-parametric, probabilistic, and interpretable characteristics. Fast-charging profiles provide a promising way to improve usage efficiency by shortening lengthy charging time. However, the random and highly dynamic fast-charging profiles will affect the availability and generality of existing health indicators and may thus make the existing methods fail to work. Moreover, the scalability of Gaussian process regression is still challenging, which suffers from cubic complexity to data size. To improve the scalability while retaining desirable health prediction quality under fast-charging scenarios, this paper proposes a scalable deep recurrent structure for battery health prognosis. First, a health indicator is extracted by applying discrete wavelet transform (DWT) on partial fast-charging profiles. Second, a recurrent structure is proposed to establish battery health prognosis models, which encapsulates the series dependency learning of a short-term memory network, while retaining the non-parametric probability advantages of Gaussian process regression. Then, structured sparse approximations and a semi-stochastic gradient procedure are established for scalable training and prediction by optimizing the Gaussian process marginal likelihood. Finally, experimental results conducted on battery aging datasets using different fast-charging profiles demonstrate the state-of-the-art performance on robustness, prediction accuracy, scalability, and predictive uncertainties.

## code description:
__partial_profiles__: 对MIT数据集根据零电流位置进行切片   

__piece_dwt__: 将切片后的partial数据经过DWT变成14个分量   

__feature_decision_two_profile__：从Pearson和Spearman相关性选择特征  

__kpg_advanced_plot__：训练LSTM-GPR、GPR、LSTM模型并估计SOH结果并对比画图  

__RUL_box__：训练LSTM-GPR、LSTM模型并估计RUL结果并对比画图  

__time_consumption_plot__：计算时间画图  

## Poster
<img width="3024" height="4536" alt="poster" src="https://github.com/user-attachments/assets/85437bc9-bdb4-4de1-a4f5-8a8b20d478d7" />

