export module rahml;

import <memory>; // for shared ptrs
import <vector>;
import <utility>;
import <unordered_set>;
import <algorithm>;
import <cmath>;


struct Node { // abstract base class
    double data;
    double grad = 0.0;
    std::vector<std::shared_ptr<Node>> inputs; // the nodes that came
    // together to form it. Ex. y = x + z, y points at x and z

    // we need to own input list, so copy input
    explicit Node(double data, 
        std::vector<std::shared_ptr<Node>> inputs = {}) : 
        data{data}, inputs{std::move(inputs)} {}
    virtual ~Node() = default;
    virtual void backward_step() = 0; // update inputs' gradients
};

struct LeafNode final : Node {
    explicit LeafNode(double data) : Node{data} {}
    void backward_step() override {} // do nothing b/c no inputs 
                                     // to send gradients too
};

struct AddNode final : Node {
    AddNode(std::shared_ptr<Node> lnode, 
        std::shared_ptr<Node> rnode) : 
        Node{lnode->data + rnode->data, {lnode, rnode}} {}
    void backward_step() override {
        // Ex. z = x + y (in this example our node is currently z)
        // ∂z/∂x = 1 (because y here would just be some constant)
        // now if a = 10 * z then ∂a/∂x = ∂a/∂z * ∂z/∂x
        // so we pass down what our current grad is too our kids
        inputs[0]->grad += grad * 1;
        inputs[1]->grad += grad * 1;
    }
};

struct NegNode final : Node {
    NegNode(std::shared_ptr<Node> node) : Node{-node->data, {node}} {}
    void backward_step() override {
        // Ex. z = -x (in this example our node is currently z)
        // ∂z/∂x = -1, now if a = 10 * z then ∂a/∂x = ∂a/∂z * ∂z/∂x
        // thus we pass down what our grad is to our kids
        inputs[0]->grad += grad * -1; // += -1 * grad
    }
};

struct MultNode final : Node {
    MultNode(std::shared_ptr<Node> lnode,
        std::shared_ptr<Node> rnode) :
        Node{lnode->data * rnode->data, {lnode, rnode}} {}
    void backward_step() override {
        // Ex. z = x * y (current node is z)
        // assume y stays constant dz/dx = y (product rule)
        // if a = 10 * z then to find how x affects a we need to use chain rule
        // and distribute to kids: ∂a/∂x = ∂a/∂z * ∂z/∂x = grad * y->data
        inputs[0]->grad += grad * inputs[1]->data;
        inputs[1]->grad += grad * inputs[0]->data;
    }
};

struct DivNode final : Node {
    DivNode(std::shared_ptr<Node> lnode, 
        std::shared_ptr<Node> rnode) :
        Node{lnode->data / rnode->data, {lnode, rnode}} {}
    void backward_step() override {
        double x_data = inputs[0]->data;
        double y_data = inputs[1]->data;
        // z = x / y (current node is z). ∂z/∂x = 1/y
        // ex. a = 10 * z so we are finding da/dx so using chain rule multiply
        // by grad
        inputs[0]->grad += grad * (1.0 / y_data);
        // ∂z/∂y = x * -1/y^2
        inputs[1]->grad += grad * (-x_data / (y_data * y_data));
    }
};

struct TanhNode final : Node {
    explicit TanhNode(std::shared_ptr<Node> node) : 
        Node{std::tanh(node->data), {node}} {}
    void backward_step() override {
        // the derivative of tanh is 1 - tanh^2(x)
        // so using the chain rule:
        inputs[0]->grad += grad * (1.0 - data * data);
    }
};

struct ReLUNode final : Node {
    explicit ReLUNode(std::shared_ptr<Node> node) :
        Node(std::max(0.0, node->data), {node}) {}
    void backward_step() override {
        // when <= 0 this function is just y = 0 which has 0 slope
        // so we don't add gradient to this. when > 0 y = x so slope of 1
        if (data > 0.0) {
            inputs[0]->grad += grad;
        }
    }
};

// append each node after its inputs. By construction, when we reverse this, it
//   lets each node receive its full gradient before passing it on to inputs.
void collect_nodes(
    const std::shared_ptr<Node> &current,
    std::unordered_set<Node *> &seen,
    std::vector<std::shared_ptr<Node>> &ordered
) {
    if (seen.contains(current.get())) {
        // already visited it in another path
        return;
    }
    seen.insert(current.get()); // so we don't add it multiple times
    for (const auto& input : current->inputs) {
        // do the dependencies first
        collect_nodes(input, seen, ordered);
    }
    ordered.push_back(current);
}

export namespace rahml {

    class Value {
        std::shared_ptr<Node> node; // multiple could point, Ex:
        /*
        z = x * y; 
        f = y + x;
        z points at x and y and so does f, thus we must use shared ptrs so
        that they don't get double deleted
        */
        Value(std::shared_ptr<Node> node) : node{std::move(node)} {}
    public:
        explicit Value(double data)
            : node{std::make_shared<LeafNode>(data)} {}
        double data() const {
            return node->data;
        }
        double grad() const {
            return node->grad;
        }
        // backpropogation:
        void backward() {
            std::unordered_set<Node *> seen;
            std::vector<std::shared_ptr<Node>> ordered;
            // find reachable nodes and make order
            collect_nodes(node, seen, ordered); // dependency order for the
                                                // operations that make them up
            // reset to 0, we want to start fresh
            for (const auto &curr : ordered) curr->grad = 0.0;

            std::reverse(ordered.begin(), ordered.end());
            node->grad = 1.0; // seed b/c derivative of itself is always 1
            // propogate
            for (const auto &curr : ordered) {
                // correctly give grads to children
                curr->backward_step();
            }
        }
        
        Value operator+(const Value &other) const {
            return Value{std::make_shared<AddNode>(node, other.node)};
        }
        Value operator+(double number) const {
            return *this + Value{number};
        }
        Value operator-() const {
            return Value{std::make_shared<NegNode>(node)};
        }
        Value operator-(const Value &other) const {
            // subtracting is the same as adding a negative number
            return *this + (-other); // redirection to using our + and -
        }
        Value operator-(double number) const {
            return *this - Value{number};
        }
        Value operator*(const Value &other) const {
            // this will get copy elided away
            return Value{std::make_shared<MultNode>(node, other.node)};
        }
        Value operator*(double number) const {
            return *this * Value{number};
        }
        Value operator/(const Value &other) const {
            return Value{std::make_shared<DivNode>(node, other.node)};
        }
        Value operator/(double number) const {
            return *this / Value{number};
        }
        Value tanh() const {
            return Value{std::make_shared<TanhNode>(node)};
        }
        Value relu() const {
            return Value{std::make_shared<ReLUNode>(node)};
        }
    };
    Value operator+(double number, const Value &value) {
        return value + number; // swap order so it calls function in Value class
    }
    Value operator-(double number, const Value &value) {
        // here the order does matter, we can't just redirect
        return Value{number} - value;
    }

    Value operator*(double number, const Value &value) {
        return value * number;
    }

    Value operator/(double number, const Value &value) {
        return Value{number} / value;
    }
}

