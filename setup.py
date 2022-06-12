import setuptools

setuptools.setup(
    name="persist",
    version="0.0.1",
    author="Ian Covert",
    author_email="icovert@cs.washington.edu",
    description="Gene probe selection for spatial transcriptomics",
    long_description="""
        Predictive and robust gene selection for spatial transcriptomics
        (PERSIST) is a computational approach for selecting a small number
        of target genes. It uses deep learning to select a set of inputs that
        can reconstruct the genome-wide expression profile, or that can predict
        a specific target variable of interest (e.g., cell type labels).
    """,
    long_description_content_type="text/markdown",
    url="",
    packages=['persist'],
    install_requires=[
        'numpy',
        'torch',
        'tqdm'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
)
