export module rahml;

import <memory>; // for shared ptrs
import <vector>;
import <utility>;

export namespace rahml {

    class Value {

        struct Node { // abstract base class
            double data;
            double grad = 0.0;
            std::vector<std::shared_ptr<Node>> inputs; // the nodes that came
            // together to form it. Ex. y = x + z, y points at x and z

            explicit Node(double data, 
                std::vector<std::shared_ptr<Node>> inputs = {}) : 
                data{data}, inputs{std::move(inputs)} {}
            virtual ~Node() = default;
            virtual void backward_step() = 0; // recursive chain rule
        };

        struct LeafNode final : Node {
            explicit LeafNode(double data) : Node{data} {}
            void backward_step() override {} // do nothing b/c no inputs 
                                            // to send gradients too
        };

        std::shared_ptr<Node> node; // multiple could point, Ex:
        /*
        z = x * y; 
        f = y + x;
        z points at x and y and so does f, thus we must use shared ptrs so
        that they don't get double deleted
        */

    public:
        explicit Value(double data)
            : node{std::make_shared<LeafNode>(data)} {}
        double data() const {
            return node->data;
        }
        double grad() const {
            return node->grad;
        }
    };

}
