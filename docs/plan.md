# rahml plan of attack

**DD1 milestone.** The terminal demo accepts two numbers, computes `(x+y)*y` using `Value`, and prints the result and both gradients. For `x=2` and `y=3`, the expected output is 15 with gradients 3 and 8 (to verify this run make and then `./build/demo 2 3`). The scalar engine also supports subtraction, division, tanh and ReLU. My regression example learns `y=2x+1` (run `make regression` to try this out). It currently creates a new pair of weight and bias nodes at each training step.
 
**Nov. 26**
- Update the regression demo so it reuses the same weight and bias each step. Add ways to reset gradients and check them numerically.

**Nov. 30**
- Build reusable layers and a trainer. Use them to train the regression and XOR models.

**Dec. 3**
- Finish batch training, CSV input, and saving and loading weights. Write a second C++ client that uses a custom layer and operation.

**Dec. 8**
- Build the n-dimensional tensor type, `Tensor<T, NumDim>`, for `float` and `double`. Add tensor operations, backpropagation and matrix multiplication with OpenBLAS.

**Dec. 11**
- Turn text into batches of context words and next-word targets. Connect the word embeddings to the network and train it on a dataset.

**Dec. 13**
- Train the word predictor and compare it with a word-frequency baseline on unseen text. Save a model the marker can run to predict the next word or generate a short passage.

**Dec. 15**
- Finish the design document and updated UML.

**Dec. 16**
- Submit to Marmoset by 11:59 p.m.
