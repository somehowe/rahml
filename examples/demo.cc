import rahml;
import <iostream>;
import <string>;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        // need 2 values (the other 1 is the program name)
        std::cerr << "Usage: " << argv[0] << " <num1> <num2>" << std::endl;
        return 1;
    }
    double x_input;
    double y_input;
    try {
        x_input = std::stod(argv[1]);
        y_input = std::stod(argv[2]);
    } catch (std::invalid_argument &) {
        std::cerr << "Usage: " << argv[0] << " <num1> <num2>" << std::endl;
        return 1;
    } catch (std::out_of_range &) {
        std::cerr << "Usage: number in incorrect range for a double\n";
        return 1;
    }
    rahml::Value x{x_input};
    rahml::Value y{y_input};
    rahml::Value result = (x + y) * y;
    result.backward();
    std::cout << "result = " << result.data() << std::endl;
    std::cout << "x.grad = " << x.grad() << std::endl;
    std::cout << "y.grad = " << y.grad() << std::endl;
}
