import setuptools

setuptools.setup(
    name="latticedefects",
    version="1.0",
    author='Alon Sardas',
    author_email='alon.sardas@mail.huji.ac.il',
    packages=['latticedefects'],
    install_requires=['numpy', 'matplotlib'],
    scripts=['latticedefects/defectsplot.py']
)
