import setuptools

requirements = [
    'numpy',
    'torch',
    'tqdm',
    'scikit-learn',
    'h5py',
    'anndata',
    'scanpy',
    'matplotlib',
    'scipy',
    'pandas',
    'toml'
]

setuptools.setup(
    name='persist',
    version='0.0.1',
    author='Ian Covert',
    author_email='icovert@cs.washington.edu',
    description='PERSIST: predictive and robust gene selection for spatial transcriptomics',
    long_description='''
        Predictive and robust gene selection for spatial transcriptomics
        (PERSIST) is a computational approach to select a small number of
        target genes for FISH experiments. PERSIST relies on a reference
        scRNA-seq dataset, and it uses deep learning to identify genes that are
        predictive of the genome-wide expression profile or any other target of
        interest (e.g., transcriptomic cell types). PERSIST binarizes gene
        expression levels during the selection process to account for the
        measurement shift between scRNA-seq and FISH expression counts.
    ''',
    long_description_content_type='text/markdown',
    url='https://github.com/iancovert/persist',
    packages=['persist'],
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
    ],
    python_requires='>=3.6',
)
