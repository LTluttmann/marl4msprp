from setuptools import setup

setup(
    name='marlprp',
    version='0.1',
    python_requires='>=3.9.0',
    # py_modules=['gurobi', 'heuristics', 'instances', 'model'],
    py_modules=[],
    install_requires=["gurobipy", "pandas", "matplotlib"],
    # entry_points={
    #     'console_scripts': [
    #         'train-actor = rl4mixed.trainer:main',
    #         'solve-exact = rl4mixed.gurobi.main:main',
    #         'sim-testdata = rl4mixed.validation:main',
    #     ],
    # },
)