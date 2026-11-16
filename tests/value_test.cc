import rahml;
import <cassert>;

using rahml::Value;

int main() {
    Value a{3};
    Value b{2};
    assert(a.data() == 3);
    assert(a.grad() == 0);
    assert(b.data() == 2);
    assert(b.grad() == 0);
    Value c = a + b;
    assert(c.data() == 5);
    assert(c.grad() == 0);
}