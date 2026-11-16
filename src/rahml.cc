export module rahml;

import <memory>; // for shared ptrs
import <vector>;
import <utility>;


struct Node { // abstract base class
    double data;
    double grad = 0.0;
    std::vector<std::shared_ptr<Node>> inputs; // the nodes that came
    // together to form it. Ex. y = x + z, y points at x and z

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
        inputs[0]->grad += grad;
        inputs[1]->grad += grad;
    }
};

export namespace rahml {

    class Value {
        std::shared_ptr<Node> node; // multiple could point, Ex:
        /*
        z = x * y; 
        f = y + x;
        z points at x and y and so does f, thus we must use shared ptrs so
        that they don't get double deleted
        */
        Value(std::shared_ptr<Node> &&node) : node{std::move(node)} {}
    public:
        explicit Value(double data)
            : node{std::make_shared<LeafNode>(data)} {}
        double data() const {
            return node->data;
        }
        double grad() const {
            return node->grad;
        }
        
        Value operator+(const Value &other) const {
            return Value{std::make_shared<AddNode>(node, other.node)};
        }
    };

}
