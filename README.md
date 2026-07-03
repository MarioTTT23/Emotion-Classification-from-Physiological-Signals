Abstract

From a constructivist perspective, emotional experiences emerge when individuals interpret their physiological states within a given context. 

Alexithymia, a condition characterized by a reduced awareness of emotions, raises a discussion: 
is this deficit caused by blunted physiological responses, or by a decoupling between physiological responses and cognitive processing? 

To investigate this, the thesis implements a novel method for measuring objective interoceptive accuracy. 
Previous attempts to measure it, involved explicitly asking subjects to count their heartbeats, and measuring how well could they do it (Garfinkel et al. 2015).
Alternatively, the current study evaluates whether their physiological states can accurately predict their emotional states, 
expecting that subjects with good interoception will report emotional states according to their physiological states.

Participants were exposed to emotion-inducing stimuli targeting Anger, Sadness, and Happiness 
Physiological responses were continuously recorded using a wristband (Empatica EmbracePlus) to capture Photoplethysmography (PPG) and Electrodermal Activity (EDA). 
Given the limited amount of data per participant; Heart Rate (Shaffer et al., 2020), Variability of Systolic Area (Paul et al., 2024), and the sample entropy of the skin conductance response (Nardelli et al., 2022) were extracted to train Support Vector Machine classifiers. 

We used the accuracy of the classifiers to quantify physiological informativeness about subjects' emotional states. 

Classification accuracies were significantly better than chance. 
When predicting subject-reported emotions, analyses showed a high probability (approximately 95%) that higher Difficulty Identifying Feelings (DIF) scores predicted lower classification accuracy. Suggesting less correlation between their 'physiological emotions' and what they report.
Conversely, when predicting stimulus-intended emotions, DIF scores were not a predictor of classification accuracy. Suggesting no blunted physiological responses, and thus, a possible physiological-cognitive decoupling.

Garfinkel, S. N., Seth, A. K., Barrett, A. B., Suzuki, K., & Critchley, H. D. (2015). Knowing your own heart: distinguishing interoceptive accuracy from interoceptive awareness. Biological psychology, 104, 65-74.

Nardelli, M., Greco, A., Sebastiani, L., & Scilingo, E. P. (2022). ComEDA: A new tool for stress assessment based on electrodermal activity. Computers in Biology and Medicine, 150, 106144.

Paul, A., Chakraborty, A., Sadhukhan, D., Pal, S., & Mitra, M. (2024). A simplified PPG based approach for automated recognition of five distinct emotional states. Multimedia Tools and Applications, 83(10), 30697-30718.

Shaffer, F., Meehan, Z. M., & Zerr, C. L. (2020). A critical review of ultra-short-term heart rate variability norms research. Frontiers in neuroscience, 14, 594880.
