# Skill Set Challenge

This project contains an implementation of the Kinetic Component Analysis (KCA) algorithm from the paper **Kinetic Component Analysis** by Marcos Lopez de Prado and Riccardo Rebonato.

# Insallation
`pip install -r requirements.txt`

# Way of Working
I work in an iterative manner.  Specifically:
- I read the paper to get an idea of what it is trying to achieve.
- I then implement the code in the paper and test that it works.
- Understand how the Kalman Filter works.
- Read KCA paper more thoroughly to understand better the purpose and implementation.
- After these initial steps then I spend a great deal of time thinking about the best use of this algorithm (Alpha Modeling) and the data that would have the characteris that would be able to match well with the algorithm.  In the case of this exercise this took the majority of the time.
- I then proceeded to collect the data.  In this case from the Bloomberg Professional platform.
- I then proceeded to build the helper modules: metrics, pipeline, plots and abstract alpha strategy class.
- Then I built the KCAMomentum and KCAMeanReversion alpha strategy.

# Desing Choices
I generally try to build my projects with an API that is easily extendable using the coding principles of low coupling and strong cohesion.  In this case it is trivial to add another alpha strategy to the project.


