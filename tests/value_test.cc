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
    const Value d = a + (a + b);
    assert(d.data() == 8);
    assert(d.grad() == 0);
    c = d + a;
    assert(c.data() == 11);
    assert(c.grad() == 0);
    assert((a + a).data() == 6);
}