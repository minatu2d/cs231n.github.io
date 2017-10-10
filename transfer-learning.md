---
layout: page
permalink: /transfer-learning/
---

(These notes are currently in draft form and under development)

Table of Contents:

- [Transfer Learning](#tf)
- [Additional References](#add)

<a name='tf'></a>

## Transfer Learning

Trong thực tế, rất ít người đi train toàn bộ Convolutional Network từ đầu (với khởi tạo ngẫu nhiên), bởi vì hiếm khi có được một dataset có kích thước. Thay vào đó, người ta rất hay pretrain một mạng ConvNet trên một dataset rất lớn(ví dụ: ImageNet, bộ dữ liệu chứa đến 1.2 triệu ảnh với 1000 category), sau đó sử dụng ConvNet đã được pretrained đó làm mạng khởi tạo hoặc bộ tác đặc trưng cố định (fixed feature extractor) cho các nhiệm vụ cụ thể hơn. Ba ngữ cảnh chính của Transfer Learning như sau:

- **ConvNet được sử dung làm bộ tách đặc trưng cố định (fixed feature extractor)**. Lấy một mạng ConvNet được pretrained trên ImageNet, bỏ lớp fully-connected cuối cùng (đầu ra của lớp này chính là score cho 1000 class cho một task khác sau đó, như ImageNet chẳng hạn), sau đó sử dụng phần còn lại của ConvNet như là một bộ tách đặc trưng cố định - fixed feature extractor trên dataset mới. Trong AlexNet, chỗ này sẽ tính ra một vector có số chiều 4096-D cho mỗi ảnh chứa activations của các lớp ẩn ngay trước khi đến classifier. Chúng ta gọi nhưng đặc trưng được tính ra đó là **CNN codes**. Một điểm quan trọng cho performance là những codes này được ReLUd (i.e. thresholded tại zero) nếu chúng được threshodled trong quá trình train ConvNet trên dữ liệu ImageNet (là một trường hợp thông thường). Một khi bạn đã extract được code 4096-D cho tất cả các ảnh rồi, thì train linear classifier (ví dụ: Linear SVM hoặc Softmax classifier) dataset mới thôi.
- **Tinh chỉnh mạng ConvNet - Fine-tuning the ConvNet**. Chiến lực thứ 2 là không chỉ thay thế và train lại classifier ở đỉnh của ConvNet cho tập dataset mới, mà nó còn fine-tune các trọng số của pretrained network bằng việc tiếp tục thực hiện phản hồi ngược - backpropagation. Người ta có thể fine-tune tất cả các lớp của ConvNet, hoặc giữa một số lớp ở đầu hay gần dữ liệu vào không đổi (do lo ngại overfitting) và chỉ fine-tune một vài vị trí ở mức cao hơn - higher-level trong mạng đó mà thôi. Điều này được áp dụng do quan sát rằng các đặc trưng sớm từ ConvNet trường tổng quan hơn (ví dụ: edge detectors - phát hiện cạnh hoặc color blob detectors - phát hiện khối màu) vì thế nó sẽ hữu ích cho nhiều mục đích khác nhau, nhưng các lớp phía sau ConvNet dần dần trở nên đặc trưng hơn cho chi tiết mỗi lớp có trong dataset ban đầu. Lấy trường hợp của ImageNet làm ví dụ, nó chứa rất nhiều giống chó, một phần quan trọng trong sức mạnh biểu diễn của ConvNet là có thể đưa ra các đặc trưng để phân biệt các giống cho khác nhau.
- **Pretrained models**. Vì các ConvNets hiện đại mất đến 2-3 tuần để train với nhiều GPUs cho ImageNet, người ta rất hay đưa ra các điểm checkpoint của mạng ConvNet cuối cùng, rồi người khác có thể sử dụng cái mạng đó cho việc fine-tuning. Ví dụ, Caffe library có cái [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) cho phép người ta chia sẻ weight cho nhau được.

**Khi nào và tinh chỉ (fine-tune) như thế nào?** Làm thế nào để bạn quyết định loại transfer learning bạn nên thực hiện trên tập dữ liệu mới? Cái này dựa trên một vài yếu tố, nhưng quan trọng hơn cả là kích thước dataset mới (nhỏ hay lớn), và tính tương đồng của nó với dataset ban đầu (ví dụ: tương tự ImageNet dựa trên nọi dung ảnh và các class, hoặc rất khác, như ảnh dưới kính hiển vi chẳng hạn). Nhớ rằng các đặc trưng lấy được từ ConvNet thì thường tổng quát ở các lớp đầu (gần ảnh) và chi tiết (gần với dữ liệu cụ thể) ở các lớp sau, dưới đây là một vài quy tắc phổ biến dựa trên kinh nghiệm để xử lý cho 4 tình huống chính:

1. *Dataset mới nhỏ và tương tự dataset ban đầu*. Vì dữ liệu nhỏ, không phải là ý tưởng tốt nếu đi  fine-tune cái ConvNet, vì dễ kéo theo những lo ngại về overfitting. Vì dữ liệu tương tự dữ liệu ban đầu, chúng ta sẽ mong muốn các đặc trưng ở mức cao hơn (higher-level features) trong ConvNet gần với tập dữ liệu mới. Bởi vậy, ý tưởng tốt nhất ở đây là train cái linear classifier dựa trên CNN codes.
2. *Dataset mới lớn và tương tự dataset ban đầu*. Vì chúng ta có nhiều dữ liệu hơn, chúng ta có thể có niềm tin rằng sẽ không bị overfit nếu chúng ta cố gắng fine-tune toàn mạng.
3. *Dataset mới nhỏ và rất khác dataset ban đầu*. Vì dữ liệu nhỏ, dường như tốt nhất là chỉ train linear classifier thôi. Vì dataset rất khác nhau, có thể không phải tốt nhất khi train classifier ở phần trên của network, cái mà chứa nhiều đặc trưng phụ thuộc mạnh vào dữ liệu. Thay vào đó, có thể sẽ chạy tốt hơn khi train SVM classifier từ các activations đâu đó trong mạng.
4. *Dataset mới lớn và rất khác dataset ban đầu*. Vì dataset rất lớn, chúng có thể mong muốn rằng chúng ta có đủ khả năng để train ConvNet từ đầu. Tuy nhiên, trong thực tế vẫn rất có nếu khi khởi tạo bằng một tập trọng số từ một pretrained model. Trong trường hợp này, chúng ta có đủ dữ liệu và có niềm tin để fine-tune toàn bộ mạng.

**Lời khuyên thực tế**. Có vài thứ cần luôn nhớ khi thực hiện Transfer Learning:

- *Ràng buộc từ pretrained models*. Chú ý rằng nếu bạn muốn sử dụng pretrained network, bạn có thể bị ràng buộc một chút về kiến trúc mạng bạn sử dụng trong cho dataset mới. Ví dụ, bạn không thể lấy ra bất cứ lớp Conv từ pretrained network được. Tuy nhiên, vài thay đổi như sau thì không vấn đề gì: do mô hình chia sẻ tham số, bạn có thể dễ dàng chạy một mạng pretrainede trên các ảnh có với có số chiều hình học khác nhau. Đây là điều hiển nhiên trong các lớp Conv/Pool vì hàm forward của chúng không phụ thuộc vào kích thước không gian khối đầu vào (miễn là stride khớp là được). Trong trường hợp các lớp FC, điều này vẫn đúng true bởi vì một lớp FC có thể chuyển đổi được sang một lớp Convolutional: ví dụ, trong AlexNet, khối pooling cuối cùng ngay trước lớp FC đầu tiên có kích thước [6x6x512]. Vì thế, lớp FC đó đang nhìn khối pool trước nó tương đương một lớp Convolutional có receptive field kích thước bằng 6x6, và khi đó padding là 0.
- *Learning rates*. Một điều rất phổ biến là sử dụng một learning rate nhỏ hơn các trọng số mạng ConvNet được fine-tuned, khi so sánh với các trọng số (được khởi tạo ngẫu nhiên) cho một linear classifier mới sử dụng để tính toán class score cho dữ liệu mới. Đó là vì, chúng ta mong muốn một trọng số ConvNet tương đối tốt, vì thế chúng ta sẽ không muốn làm méo chúng quá nhanh và quá nhiều lần (đặc biệt so với trường hợp Linear Classifier mới được học từ những giá trị khởi tạo ngẫu nhiên).

<a name='tf'></a>

## Additional References

- [CNN Features off-the-shelf: an Astounding Baseline for Recognition](http://arxiv.org/abs/1403.6382) trains SVMs on features from ImageNet-pretrained ConvNet and reports several state of the art results.
- [DeCAF](http://arxiv.org/abs/1310.1531) reported similar findings in 2013. The framework in this paper (DeCAF) was a Python-based precursor to the C++ Caffe library.
- [How transferable are features in deep neural networks?](http://arxiv.org/abs/1411.1792) studies the transfer learning performance in detail, including some unintuitive findings about layer co-adaptations.
