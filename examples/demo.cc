import rahml;
import <iostream>;

int main() {
    rahml::Value x{2.0};
    rahml::Value copy = x;
    
    std::cout << "x: data=" << x.data() << 
        ", grad=" << x.grad() << std::endl;
    std::cout << "copy: data=" << copy.data() <<
        ", grad=" << copy.grad() << std::endl;
}
