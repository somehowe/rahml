import rahml;
import <iostream>;

int main() {
    rahml::Value x{2.0};
    rahml::Value y{3.0};
    rahml::Value result = (x + y) * y;
    result.backward();
    std::cout << "result = " << result.data() << std::endl;
    std::cout << "x.grad = " << x.grad() << std::endl;
    std::cout << "y.grad = " << y.grad() << std::endl;
}
