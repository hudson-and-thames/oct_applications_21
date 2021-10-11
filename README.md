# Skill Set Challenge!
[Hudson & Thames](https://hudsonthames.org/) has provided the following skillset challenge to allow potential researchers to gauge if they have the required skills to take part in the [apprenticeship program](https://hudsonthames.org/apprenticeship-program/).

Our previous cohorts built the [ArbitrageLab](https://hudsonthames.org/arbitragelab/) python library. You will be working on advanced algorithms for the MlFinLab and the PortfolioLab packages.

<div align="center">
  <img src="https://raw.githubusercontent.com/hudson-and-thames/oct_applications_21/master/images/mlfinlab_logo.png" height="250"><br>
  <img src="https://raw.githubusercontent.com/hudson-and-thames/oct_applications_21/master/images/portfoliolab_logo.png" height="250"><br>
</div> 

## Your Mission:
The following assignment is an opportunity for you to highlight your skillset and show us what you are made of! It tests your ability to implement academic research for the broader quantitative finance community, and to do it in style!

### Briefing

<div align="center">
  <img src="https://raw.githubusercontent.com/hudson-and-thames/oct_applications_21/master/images/KCA_Signals.PNG" height="300"><br>
</div>  

Read the following paper: [Kinetic Component Analysis](https://ssrn.com/abstract=2422183) *by* Marcos Lopez de Prado *and* Riccardo Rebonato. 

In a Jupyter Notebook (python):

1. Download and save a preferred set of futures (use 5-10 futures) (Can use Yahoo finance to get the data. Checkout the [yfinance](https://github.com/ranaroussi/yfinance) package.) (Else you can use [Polygon](https://polygon.io/) to download data or use your own dataset)
1. Use the implementation of the KCA from Appendice A.1.
1. Make a function to use the output from the KCA to generate a set of trading signals (Any simple trading algorithm logic).
1. Optional: Also do the above for the FFT and LOWESS algorithms (Appendices A.2. - A.4.).
1. Construct trading strategy class(es) using the code from the previous steps for the end-user to make use of (Separate classes for KCA, FFT, and LOWESS, if the last two are being added). The class should take a set of observations as input and, based on a set of parameters, generate a trading signal for the next observation (or multiple).
1. Make sure to add docstrings and follow PEP8 code style checks. Have plenty of inline comments, good variable names and don't overcomplicate things unnecessarily. It should be easy for the user to make use of.
1. Showcase your new KCA Strategy (and FFT, LOWESS, if added) in a Jupyter Notebook, apply to the downloaded dataset from p. 1. and analyse performance, make conclusions, add good visualisations.
1. Add an introduction, body, and conclusion showcasing your new implementation. (Use the correct style headers)
1. Make a Pull Request to this repo so that we can evaluate your work. (Create a new folder with your name)
1. Bonus points if you add unit tests (in a separate .py file).
1. Provide a write-up explaining your way-of-work, your design choices, maybe a UML diagram, and learnings.
1. Deadline: 22nd of October 2021  

### How Will You be Evaluated?

Being a good researcher is a multivariate problem that can't be reduced to a few variables. You need to be good at, mathematics, statistics, computer science, finance, and communication.

Pay close attention to detail and know that code style, inline comments, and documentation are heavily weighted.

The one consistent theme between all of our top researchers is their ability to go above and beyond what is asked of them. Please take the initiative to highlight your talent by going the extra mile and adding functionality that end-users would love!

### Notes
* Your code for the implementation should be contained in a .py file that you import into your notebook. Please don't have large chunks of code in your notebook.
* IDE Choice: Pycharm, NOT Spyder.
* Save your data with your PR so that we can evaluate it.
* Turn to the previous cohorts' submissions: [1](https://github.com/hudson-and-thames/oct_applications), [2](https://github.com/hudson-and-thames/march_applications_21), [3](https://github.com/hudson-and-thames/june_applications_21) for inspiration.

## Institutional - Need to Know

<div align="center">
  <img src="https://raw.githubusercontent.com/hudson-and-thames/oct_applications_21/master/images/logo_black_horisontal.png" height="150"><br>
</div>

* **Company Name**: Hudson and Thames Quantitative Research
* **Company Brief**: Our core focus is on the implementation of research within buy-side asset management.
* **Company Website**: [https://hudsonthames.org/](https://hudsonthames.org/)
* **Locked Achievement**: Apprenticeship Program Alumni
* **Location**: Virtual Team (We are all in different time zones across the world.)
* **Education**: Familiarity with machine learning, statistics, and applied maths. We care a lot more about what you can do rather than your exact qualifications.

### Day on Day Activity
* Implement academic research for machine learning in finance.
* Python
* Unit tests
* PEP8
* Continuous integration
* Documentation
* Writing articles
* Public Speaking

### Skills:
* Must speak fluent English. There is a documentation requirement so English is an absolute requirement.
* Python
* Machine Learning
* Software engineering
* Object Orientated Programming
* Linear Algebra
