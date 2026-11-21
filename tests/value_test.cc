import rahml;
import <cassert>;
import <iostream>;

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

    // let's test mult now
    {
        Value x{4.0};
        Value y{-2.0};
        Value prod = x * y;
        assert(prod.data() == -8.0);
        assert(prod.grad() == 0.0);
        prod.backward();
        assert(prod.grad() == 1.0);
        assert(x.grad() == -2.0);
        assert(y.grad() == 4.0);
    }
    // repeated input, make sure we are adding to the gradients
    {
        Value x{3.0};
        Value sqr = x * x;
        assert(sqr.data() == 9.0);
        sqr.backward();
        assert(x.grad() == 6.0); // derivative of x^2 = 2x = 2 * 3
    }
    // zero case
    {
        Value x{0.0};
        Value y{-2.0};
        Value prod = x * y;
        assert(prod.data() == 0.0);
        assert(prod.grad() == 0.0);
        prod.backward();
        assert(prod.grad() == 1.0);
        assert(x.grad() == -2.0);
        assert(y.grad() == 0.0);
    }
    // Nov. 20 goal that we had
    {
        Value x{2.0};
        Value y{3.0};
        Value result = (x + y) * y;
        assert(result.data() == 15.0); // 2 + 3 = 5 * 3 = 15
        result.backward();
        // y*x + y^2
        assert(x.grad() == 3.0); // dr/dx = y
        assert(y.grad() == 8.0); // dr/dy = x + 2y = 2 + 6 = 8
        // repeat to make sure grads replaced
        result.backward();
        assert(x.grad() == 3.0);
        assert(y.grad() == 8.0);
    }
    // make sure collection still working and that we basically
    //   calculate gradients at shared node before passing to inputs 
    {
        Value x{2.0};
        Value y = x * x;
        Value result = y * x + y;
        assert(result.data() == 12.0);
        result.backward();
        assert(y.grad() == 3.0); // dr/dy = x + 1
        assert(x.grad() == 16.0); // res = y * x + y = x^3 + x^2 = 3x^2 + 2x
    }

    // now we test negation and subtraction
    // negating reverse the gradient
    {
        const Value x{3.0};
        Value result = -x;
        assert(result.data() == -3.0);
        result.backward();
        assert(x.grad() == -1.0);
    }
    // Subtraction will give negative sign for derivative
    {
        Value x{5.0};
        const Value y{2.0};
        Value result = x - y;
        assert(result.data() == 3.0);
        result.backward();
        assert(x.grad() == 1.0);
        assert(y.grad() == -1.0);
    }
    // Contributions cancel when both are same node (1 - 1 = 0)
    {
        Value x{4.0};
        Value result = x - x;
        assert(result.data() == 0.0);
        result.backward();
        assert(x.grad() == 0.0);
    }
    // Case where negation and subtraction combined
    {
        Value x{3.0};
        Value y{1.0};
        // -3 * (3 - 1) = -3 * 2 = -6
        // -x^2 + xy
        // ∂r/∂x = -2x + y = -6 + 1 = -5
        // ∂r/∂y = x = 3.0
        Value result = (-x) * (x - y);
        assert(result.data() == -6.0);
        result.backward();
        assert(x.grad() == -5.0);
        assert(y.grad() == 3.0);
    }
    // test division
    {
        Value x{8.0};
        Value y{2.0};
        Value result = x / y;
        assert(result.data() == 4.0);
        result.backward();
        assert(x.grad() == 0.5);
        assert(y.grad() == -2.0);
    }
    // fraction case
    {
        Value x{1.5};
        Value y{0.5};
        Value result = x / y;

        assert(result.data() == 3.0);
        result.backward();
        assert(x.grad() == 2.0);
        assert(y.grad() == -6.0);
    }

    // dividing by self
    {
        Value x{4.0};
        Value result = x / x;

        assert(result.data() == 1.0);
        result.backward();
        assert(x.grad() == 0.0);
    }

    // make sure it adds both contributions when multiplying (accumulation)
    {
        Value x{8.0};
        Value y{2.0};
        Value r = x / y;
        Value result = r * r;

        assert(result.data() == 16.0);
        result.backward();
        assert(r.grad() == 8.0);
        assert(x.grad() == 4.0);
        assert(y.grad() == -16.0);
    }
}