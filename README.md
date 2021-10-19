Overview
---------

The repository contains the Odia-Santali dialect detection data i.e. text written in Odia and Santali (using Odia script). The code is also available for detecting Odia and Santali.  


Dependency
----------

Python 3.6

PyTorch (torch=1.0.1, torchtext=0.4.0, torchvision=0.4.0)

Utilities (Skopt, sklearn, numpy, Zipfile, Pandas, Pickel) 

How to Run ?
-------------
The language detection code supports both CPU/GPU. For running in CPU/GPU mode, enable/disable the "is_gpu" flag to "True" or "False" inside the "lang_detect_santali.py" file. 



Reference Paper
---------------

[1] Parida, S., ú Villatoro-Tello, E., Kumar, S., Motlicek, P., & Zhan, Q. (2020). Idiap Submission to Swiss-German Language Detection Shared Task. In Proceedings of the 5th Swiss Text Analytics Conference (SwissText) & 16th Conference on Natural Language Processing (KONVENS).

[2] Le, L., Patterson, A., & White, M. (2018). Supervised autoencoders: Improving generalization performance with unsupervised regularizers. In Advances in Neural Information Processing Systems (pp. 107-117).

[3] Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical bayesian optimization of machine learning algorithms. In Advances in neural information processing systems (pp. 2951-2959).

Contributor
------------
- Sunil Sahoo
- Brojo Kishore Mishra
- Shantipriya Parida
- Satya Ranjan Dash
- Jatindra Nath Besra
- Esau Villatoro-Tello 

Citation
--------

If you found our research helpful or influential please consider citing

@inproceedings{santali_dialect_detection,
  title={Automatic Dialect Detection for Low Resource Santali Language},
  author={Sahoo, Sunil and Mishra, Brojo Kishore and Parida, Shantipriya and Dash, Satya Ranjan and Besra, Jatindra Nath and {\'u} Villatoro-Tello, Esau},
  booktitle={Proceedings of the 19th OITS International Conference on Information Technology},
  year={2021}
}
