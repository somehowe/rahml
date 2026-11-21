# rahml: proposed additional work

**CS246E Self-Proposed Project | Ryan Howe**

I want to add tensor support and a word prediction model to rahml as additional work toward recovering the 10-percentage-point project deduction associated with transferring the midterm weight to the project. Tensor support is needed so it can be possible to make the word predicition model (as just doing values would be so slow it wouldn't be possible to train in a reasonable amount of time). I estimate this to take around 50-100 additional hours.

## 1. What I propose to add

### Tensor operations and differentiation

Currently, each `Value` represents one number. I would add `Tensor<T>`, with both `float` and `double`. I would support matrices first (potentially later going to n-dimensional tensor with template like `Tensor<T, NumDim>`) and single-element losses, with contiguous storage and explicit shapes (basically just a straight vector like [10, 20, 30, 40, 50, 60] and then a vector like [2, 3] which tells us the shape).

I would use OpenBLAS library for CPU optimized matrix multiplication. I would implement the tensor interface, shape checking, computation graph, and backward rules, along with the elementwise operations, reductions, activations, and bias broadcasting needed by the model. Matrix multiplication’s backward rule would also use OpenBLAS. Each tensor operation would create one graph node, rather than separate nodes for every individual scalar calculation.

There are some new gradient problems here. For example, a bias used by every example in a batch must receive the sum of those contributions. An embedding lookup (lookup table that maps a word to a vector) must also add gradients when the same word occurs more than once (bad to reset because then we would lose knowledge that the word occured multiple times). I would test these cases, incorrect shapes, and shared intermediate tensors, and compare small examples against the scalar engine and numerical derivatives.

### A word prediction model

I want to train a model that takes several previous **words** and predicts the next word. For example, the terminal demo could accept a prompt, show the five most likely next words, and generate a longer continuation by repeatedly predicting another word.

I would start with a fixed-context neural language model: learned word embeddings, a hidden layer, and an output score for each word in the vocabulary. The model would follow the general approach in [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). I would implement the forward pass, loss, gradients, and updates through rahml.

The additional work would include:

- Word tokenization, a vocabulary built from training text, and an unknown-word token.
- Embedding lookup and its backward rule, including repeated word IDs.
- Numerically stable softmax/cross-entropy for next-word classification.
- Training and evaluation on separate text splits, plus seeded sampling.
- Saving the vocabulary and context settings with the model so another process can load it and predict words.

I would use a subset of [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext). I would start out trying a vocabulary of 2,000–5,000 words, contexts of 4–8 words, and a small hidden layer, then record the actual configuration and training time.

I would evaluate as well using cross-entropy (scores the word that came next aka negative log likelihood) and perplexity (the exponential of the average cross-entropy), along with the unknown-word rate. My goal would be to get lower cross-entropy than a unigram baseline, which predicts words using only their training frequencies. Both would use the same vocabulary and unknown-word mapping. I would also show a few predictions and a short passage the model generates, and check that reloading the saved model gives the same predictions.

### Example client code and abstraction

Here is a client example. A client could add their own activation by deriving from the module interface and supplying the forward calculation and its derivative (so that we can calculate gradients on backward pass):

```cpp
namespace rt = rahml::tensor;
using Tensor = rt::Tensor<float>;

class Softsign final : public rt::Module<float> {
public:
    Tensor forward(const Tensor& x) override {
        // will apply same function to each number and return that tensor
        return rt::elementwise(x,
            // v/(1+|v|)
            [](float v) { return v / (1.0f + std::abs(v)); },
            // 1/(1+|v|)^2 is the derivative
            [](float v) {
                float d = 1.0f + std::abs(v);
                return 1.0f / (d * d);
            });
    }
    // no trainable parameters (no weights that we can update)
    std::vector<Tensor> parameters() const override { return {}; }
};
```

Inside the client program, this activation could be used alongside the built-in layers:

```cpp
// this code builds a next word predictor and then trains it

// assemble the model
rt::ContextEmbedding<float> embeddings{vocab.size(), 32, rng}; // maybe a 
// a 2000 by 32 matrix with random weights starting out.
rt::Sequential<float> head; // turns embeddings into predicitions
// next 3 build the structure
head.emplace<rt::Linear<float>>(context * 32, 128, rng);
head.emplace<Softsign>(); // adds nonlinearity
head.emplace<rt::Linear<float>>(128, vocab.size(), rng);

auto predict = [&](const rt::TokenBatch& ids) {
    auto x = embeddings.forward(ids); // this is the lookup word -> vector
    return head.forward(x); // pass that vector through the layers
};
// word_windows could be a view and maybe have an iterator that advances both
// left and right by 1
// (where examples come from, 32 examples per batch, how to pack the examples) ->
// which will return input vector and targets for what we want each to hit
auto batches = rt::batch_view(word_windows, 32, rt::TokenBatchEncoder{});
auto parameters = rt::unique_parameters(embeddings, head);
rt::CrossEntropy<float> loss;
rt::SGD<float> optimizer{parameters, 0.05f}; // changes parameters
rt::Trainer trainer{predict, loss, optimizer};
trainer.fit(batches, 20); // 20 passes through through our training loop
```

I would use this client program to train the next-word model on a subset of WikiText-2 using rahml’s tensor operations. For each batch, rahml would compute the cross-entropy loss and backpropagate through the model so SGD (basically the training) could update the word embeddings and the linear layers’ weights and biases. Sequential would own its layers through unique_ptr<Module<float>> and call forward on each one in order, so a client could insert Softsign between two built-in layers with a single emplace call.

## 2. Possible additions if more work is needed

Two other directions I could go in:

**a small Python-like expression language.** A user could write `prediction = tanh(x @ weights + bias)` in a text file. I would build a parser that reads the expression and determines the order of operations. In this example, rahml would multiply x and weights using matrix multiplication, add bias, apply tanh, and name the result prediction. I could turn it into AST, users could give their own tensors and functions. I could also make errors specific (ex. missing closing parenthesis.

**a non-trivial [PyTorch](https://github.com/pytorch/pytorch) open source contribution.** The idea is mine is in some ways a mini version of theirs so I would have a good amount of context to be able to maybe contribute. I would look for an appropriate tensor/autograd or C++ issue, and submit a PR or multiple. This would be separate from my project, but I could give like screenshots to show it got accepted for my demo.
