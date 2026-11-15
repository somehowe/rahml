export module rahml;

import <memory>; // for shared ptrs


namespace rahml::detail {

    struct Node { // abstract base class
        double data;
        double grad = 0.0;

        explicit Node(double data) : data{data} {}
        virtual ~Node() = default;
        virtual void backward_step() = 0;
    };

    struct LeafNode final : Node {
        explicit LeafNode(double data) : Node{data} {}
        void backward_step() override {} // do nothing b/c no inputs 
                                         // to send gradients too
    };

}

export namespace rahml {

    class Value {
        std::shared_ptr<detail::Node> node_;
    public:
        explicit Value(double data)
            : node_{std::make_shared<detail::LeafNode>(data)} {}
        double data() const {
            return node_->data;
        }
        double grad() const {
            return node_->grad;
        }
    };

}
