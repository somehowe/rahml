export module rahml;

import <memory>; // for shared ptrs

export namespace rahml {

    class Value {

        struct Node { // abstract base class
            double data;
            double grad = 0.0;

            explicit Node(double data) : data{data} {}
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
