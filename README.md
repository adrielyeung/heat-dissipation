# heat-dissipation
(Project for Computational Physics course in autumn 2017)

A microprocessor in a computer produces a lot of heat during operation. It is hence required to design microprocessors with appropriate heat dissipation structures. The simplest cases of a heat sink and fins are considered here, using a simplified heat transport equation.

Program written in Python. ```meshclass.py``` contains the class to create a single meshgrid object (which contains temperature data at every grid point in each component of the microprocessor), while ```multimesh.py``` contains the class to combine these into a single large meshgrid (to combine the microprocessor, heat sink and / or fins). The test cases are all contained in ```test_jacobi.py```, including (1) no heat sink, (2) with heat sink, and (3) forced convection (with heat sink and fins).

## Improvements
Too many loops and if statements. Use of ```lambda``` functions and ```np.where``` would immensely improve the situation and make the code run much quicker.

## Acknowledgements
Credits to Prof. Yoshi Uchida and Dr. Pat Scott for teaching the course, Dr. Diego Alonso Álvarez for designing the project, help along the way and feedback at the end. Also special thanks to my friend Rachel for completing the project with me all along.
