import os
import setuptools

# 如果readme文件中有中文，那么这里要指定encoding='utf-8'，否则会出现编码错误
with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as readme:
    README = readme.read()

# 允许setup.py在任何路径下执行
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setuptools.setup(
    name="general-ml-darkn",  # 库名, 需要在pypi中唯一
    version="1.0",  # 版本号
    author="Darkn Lxs",  # 作者
    author_email="1187220556@qg.com", # 作看都将（方便使用索类现问图后成我我们）
    description="用于机器学习相关的通用包",  # 简介
    long_description="见readme",  # 详细描述（一般会写在README.md中）
    long_description_content_type="text/markdown",  # README.md中描述的语法（一般为markdown)
    # url="https://github.com/pypa/sampleproject",  # 库/调目主页， 一般我们把项目托管在GitHub，放该项目的GitHub地址即可
    packages=setuptools.find_packages(),  # 默认值即可，这个是方便以后我们给库拓展新功能的
    classifiers=[  # 指定该库依赖的Python版本、license、操作系统之类的
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[  # 该库需要的依前库
        'joblib>=0.17.0',
        'pandas>=1.1.3',
        'tqdm>=4.50.2',
        'matplotlib>=3.3.2',
        'numpy>=1.19.2',
        'xgboost>=1.4.2',
        'scikit_uplift>=0.3.2',
        'scikit_learn>=0.24.2',
        'seaborn>=0.11.2',
        'scipy>=1.4.1',
        'lightgbm>=3.2.1',
        'torch>=1.5.0'
    ],
    python_requires='>=3.6',
)

# python setup.py sdist bdist_wheel