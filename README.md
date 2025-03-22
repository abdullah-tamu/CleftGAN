# CleftGAN
A StyleGAN-based generator for human faces having repaired/unrepaired cleft lip anomaly described in [CleftGAN: Adapting A Style-Based Generative Adversarial Network To Create Images Depicting Cleft Lip Deformity](https://arxiv.org/abs/) by Abdullah Hayajneh, Erchin Serpedin, Mohammad Shaqfeh, Graeme Glass and Mitchell A. Stotland.

# Requirements
Same requirements as StyleGAN3. Please refer to 
https://github.com/NVlabs/stylegan3 
for the requrements.
Also, please install opencv

# Usage:
python main.py --nimg=100 --output_path='./faces' --show_images=True

# Citation



If you find this implementation helpful in your research, please also consider citing:
```
@article{hayajneh2023unsupervised,
  title={Unsupervised anomaly appraisal of cleft faces using a StyleGAN2-based model adaptation technique},
  author={Hayajneh, Abdullah and Shaqfeh, Mohammad and Serpedin, Erchin and Stotland, Mitchell A},
  journal={Plos one},
  volume={18},
  number={8},
  pages={e0288228},
  year={2023},
  publisher={Public Library of Science San Francisco, CA USA}
}
```
