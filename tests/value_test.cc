import rahml;
import <cassert>;

using rahml::Value;

int main() {
    Value a{3.0};
    Value b{2.0};
    assert(a.data() == 3.0);
    assert(a.grad() == 0.0);
    assert(b.data() == 2.0);
    assert(b.grad() == 0.0);
    Value c = a + b;
    assert(c.data() == 5.0);
    assert(c.grad() == 0.0);
    const Value d = a + (a + b);
    assert(d.data() == 8.0);
    assert(d.grad() == 0.0);
    c = d + a;
    assert(c.data() == 11.0);
    assert(c.grad() == 0.0);
    assert((a + a).data() == 6.0);
    // test backward() method
    Value left{4.0};
    Value right{1.0};
    Value total = left + right;
    total.backward();
    assert(left.grad() == 1.0 && right.grad() == 1.0);
    // make sure that grads accumulate
    Value input{4.0};
    Value twice = input + input;
    twice.backward();
    assert(input.grad() == 2.0);
    // make sure that the order the grads are being distributed is right
    Value source{3.0};
    Value branch = source + source;
    Value loss = (branch + source) + branch;
    loss.backward();
    assert(source.grad() == 5.0);
    // repeat the call again (reset to 0's and go through the whole thing again)
    loss.backward();
    assert(source.grad() == 5.0);
}