import rahml;
import <iostream>;
import <vector>;

using rahml::Value;

int main() {
    // we are trying to get our model to predict y = 2x + 1
    const std::vector<double> xs{-2.0, -1.0, 0.0, 1.0, 2.0};
    Value w{0.0}; // weight
    Value b{0.0}; // bias
    Value total_loss{0.0};
    for (double x_num : xs) {
        Value x{x_num};
        Value target{2.0 * x_num + 1.0};
        Value prediction = w * x + b;
        Value error = prediction - target;
        total_loss = total_loss + error * error; // add it on to running sum
    }
    // loss is the mean squared error
    Value loss = total_loss / Value{static_cast<double>(xs.size())};
    loss.backward(); // run backprop

    std::cout << "loss = " << loss.data() << '\n';
    // what will happen to loss if we slightly increase w or b:
    std::cout << "w.grad = " << w.grad() << '\n';
    std::cout << "b.grad = " << b.grad() << '\n';
}