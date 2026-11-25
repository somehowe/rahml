export module rahml;

import <memory>; // for shared ptrs

struct Node;

export namespace rahml {

    template <typename Function>
    double central_difference(const Function &f, double x) {
        constexpr double h = 1e-5;
        return (f(x + h) - f(x - h)) / (2.0 * h);
    }

    class Value {
        std::shared_ptr<Node> node; // multiple could point, Ex:
        /*
        z = x * y; 
        f = y + x;
        z points at x and y and so does f, thus we must use shared ptrs so
        that they don't get double deleted
        */
        Value(std::shared_ptr<Node> node);
    public:
        explicit Value(double data);
        double data() const;
        double grad() const;
        void set_data(double new_data);
        void zero_grad();

        void backward();

        Value operator+(const Value &other) const;
        Value operator+(double number) const;
        Value operator-() const;
        Value operator-(const Value &other) const;
        Value operator-(double number) const;
        Value operator*(const Value &other) const;
        Value operator*(double number) const;
        Value operator/(const Value &other) const;
        Value operator/(double number) const;
        Value tanh() const;
        Value relu() const;
    };

    Value operator+(double number, const Value &value);
    Value operator-(double number, const Value &value);
    Value operator*(double number, const Value &value);
    Value operator/(double number, const Value &value);
}

