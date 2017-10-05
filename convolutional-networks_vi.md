---
layout: page
permalink: /convolutional-networks/
---

Table of Contents:

- [Architecture Overview](#overview)
- [ConvNet Layers](#layers)
  - [Convolutional Layer](#conv)
  - [Pooling Layer](#pool)
  - [Normalization Layer](#norm)
  - [Fully-Connected Layer](#fc)
  - [Converting Fully-Connected Layers to Convolutional Layers](#convert)
- [ConvNet Architectures](#architectures)
  - [Layer Patterns](#layerpat)
  - [Layer Sizing Patterns](#layersizepat)
  - [Case Studies](#case) (LeNet / AlexNet / ZFNet / GoogLeNet / VGGNet)
  - [Computational Considerations](#comp)
- [Additional References](#add)

## Convolutional Neural Networks (CNNs / ConvNets)

Convolutional Neural Networks tương tự  Neural Networks được nói đến ở chương trước: chúng được cấu tạo từ nhiều neurons có weights và biases có khả năng học hỏi. Mỗi neuron nhận một vài đầu vào, thực hiện phép nhân vô hướng trong ma trận và có thể kèm theo một hàm không liên tục. Toàn bộ network vẫn diễn tả một hàm khả vi để tính score: từ dữ liệu đầu vào là các điểm ảnh ở một đầu đến score cho mỗi lớp ở đầu còn lại của mạng. Chúng ta vẫn có hàm mất mát - loss function (ví dụ: SVM/Softmax) ở lớp cuối cùng (fully-connected) và tất cả mọi thứ chúng ta đã học về Neural Networks vẫn được ứng dụng.

Thế thì có gì khác ở đây? Kiến trúc ConvNet đưa ra một giả định rõ ràng rằng đầu vào luôn là ảnhs, cho phép chúng ta thực hiện những đặc tính nhất định trên kiến trúc này. Những cái này sẽ cho phép tạo ra một hàm forward function hiệu quả hơn để hiện thực
và giảm một lượng lớn các tham số trong mạng.

<a name='overview'></a>

### Architecture Overview

*Bài cũ: Neural Nets thông thương - Regular Neural Nets.* Như đã thấy ở chương trước, Neural Networks nhận một đầu vào (là một vector), chuyển đổi nó thông qua một loạt các *lớp ẩn*. Mỗi lớp ẩn có một tập các neurons, mỗi neuron được kết nối đầy đủ - tức là được nối đến toàn bộ neuron của lớp ngay trước nó, các neuron trong một lớp có chức năng hoàn toàn độc lập với nhau, chúng không chia sẻ bất cứ kết nối nào hết. Lớp kết nối cuối cùng (vẫn là kết nối đầy đủ) được gọi là "lớp đầu ra" và nếu trong bài toán phân loại nó sẽ biểu diễn score của các lớp.

*Neural Nets thông thường không thể mở rộng/hoặc thu nhỏ theo ảnh được*. Trong CIFAR-10, các ảnh đều có cùng kích thước là 32x32x3 (32 wide, 32 high, 3 color channels), chỉ một neuron được kết nối đầy đủ trong lớp ẩn đầu tiên của mạng Neural thông thường đã có đến 32\*32\*3 = 3072 weights rồi. Con số này có vẻ vẫn xử lý được, nhưng rõ ràng là cấu trúc kết nối đầy đủ này không phù hợp được với các ảnh lớn hơn. Ví dụ, một ảnh dễ nhìn hơn , giả sử là 200x200x3 chẳng hạn, sẽ dấn đến số lượng trọng số cho 1 neuron cũng đã là 200\*200\*3 = 120,000 weights rồi. Hơn nữa, thông thường chúng ta muốn có một vài neurons, vì thế số lượng weights sẽ tăng lên nhanh chóng! Vâng, rõ ràng là, kết nối đầy đủ kiểu này thì hơi phí và một lượng lớn tham số còn dẫn đến nhanh chóng bị overfitting.

*Cấu trúc 3D của các neuron - 3D volumes of neurons*. Convolutional Neural Networks tận dụng đặc tính chỉ có ảnh của đầu vào, và đem lại cho kiến trúc này sự hợp lý hơn. Cụ thể là, không giống với Neural Network thông thường, các lớp của ConvNet có các neuron được xếp ở dạng 3 chiều: **rộng, cao, sâu**. (chú ý là cái từ *sâu - depth* ở đây muốn nói đến chiều thứ 3 của khối kích hoạt, chứ không liên quan gì đến độ sâu của mạng Neural nha, hay co số biểu diễn tổng số lớp trong một mạng.) Ví dụ, ảnh đầu vào là CIFAR-10 là đầu vào của khối kích hoạt chẳng hạn, khối này tương ứng sẽ có kích thước 32x32x3 (lần lượt theo thứ tự rộng, cao, sâu). Nhưng chúng ta sẽ sớm thấy rằng, các neuron trong một lớp chỉ kết nối đến một vùng neuron nhỏ của lớp trước đó mà thôi, chứ không phải toàn bộ neuron như dạng kết nối đầy đủ - fully-connected manner. Hơn nữa, lớp đầu ra cuối cùng cho CIFAR-10 sẽ có số chiều là 1x1x10, bởi vì ở cuối của kiến trúc ConvNet chúng ta sẽ giảm ảnh ban đầu xuống còn một vector chứa score của các lớp cần phân loại mà thôi, hay được xem như là chiều độ sâu. Dưới đây là hình ảnh mô phỏng:

<div class="fig figcenter fighighlight">
  <img src="/assets/nn1/neural_net2.jpeg" width="40%">
  <img src="/assets/cnn/cnn.jpeg" width="48%" style="border-left: 1px solid black;">
  <div class="figcaption">Phía trái: Một mạng Neural thông thường, có 3 lớp. Phía phải: Một ConvNet xếp các neuron của nó ở dạng 3 chiều (rộng, cao, sâu), là một trong số các lớp. Mỗi lớp của ConvNet biến một đầu vào 3 chiều thành đầu ra cũng 3 chiều. Trong ví dụ này, lớp màu đỏ chứa dữ liệu ảnh, chiều rộng và chiều cao của nó tương ứng là của ảnh luôn, và chiều sâu là 3 (tức là 3 kênh màu Đỏ, Xanh lá, Xanh da trời).</div>
</div>

> Một ConvNet bao gồm nhiều lớp. Mỗi lớp có một API đơn giản: Biến đổi dạng ba chiều ở đầu vào sang dạng 3 chiều ở đâu ra bằng các hàm khả vi có thể có hoặc không có nhiều tham số.

<a name='layers'></a>

### Các lớp được sử dụng trong ConvNets

Như đã nói ở trên, một ConvNet đơn giản là một dãy các lớp, và mỗi lớp của ConvNet biễn một khối dữ liệu thành một khối khác thông qua một hàm khả vi của nó. Chúng ta sử dụng ba loại lớp trong kiến trúc ConvNet: **Convolutional Layer**, **Pooling Layer**, and **Fully-Connected Layer** (chính xác là cái bạn thấy trong một Neural Networks thông thường). Chúng ta xếp các lớp này lại với nhau để tạo kiến trúc ConvNet **architecture** hoàn chỉnh. 

*Kiến trúc ví dụ: Khái quát*. Chúng ta sẽ xem chi tiết bên dưới, nhưng một ConvNet đơn giản cho phân loại dữ liệu CIFAR-10 có thể có kiến trúc như sau [INPUT - CONV - RELU - POOL - FC]. Chi tiết hơn mỗi lớp:

- INPUT [32x32x3] chữa giá trị của mỗi điểm ảnh trong ảnh ban đầu, trong trường hợp này (CIFAR-10) mỗi ảnh có chiều rộng 32, chiều cao 32, và chiều sâu chính là 3 kênh màu R,G,B.
- CONV thực hiện tính toán đầu ra của các neurons, mà mỗi cái chỉ được kết nối với một vùng cục bộ ở đầu vào, phép tính toán cho neuron là tích vô hướng giữa trọng số và một vùng nhỏ mà neuron được kết nối trên khối đầu vào. Toàn bộ kết quả sẽ có kích thước [32x32x12] nếu chúng ta quyết định sử dụng 12 filter.
- RELU sẽ thực hiện một hàm activation cho mỗi phần tử, như hàm \\(max(0,x)\\) có ngưỡng là 0. Hàm này sẽ không làm thay đổi kích thước đầu vào của nó (tức vẫn là [32x32x12]).
- POOL sẽ thực hiện việc giảm lượng lấy mẫu trên chiều không gian hình học (chiều rộng, chiều cao), kết quả đầu ra sẽ cho kích thước [16x16x12].
- FC (i.e. fully-connected) sẽ tính toán core cho mỗi lớp, kết quả sẽ có kích thước [1x1x10], mỗi số trong dãy 10 số tương ứng với 1 class cần phân loại, chính là 10 mục của CIFAR-10. Đây chính là Neural Networks thông thường cũng giống như cái tên của nó,mỗi neuron trong lớp này được kết nối với toàn bộ neuron của lớp trước nó.

Như cách ở trên, ConvNets biến đổi ban đầu qua từng lớp để cuối cùng ra score cho mỗi class. Chú ý là, một vài lớp có tham số, một vài lớp thì lại không. Cụ thể là, các lớp CONV/FC thực hiện biến đổi thì, các hàm được sử dụng không chỉ là hàm activations cho dữ liệu vào của lớp đó, nó cũng chứa tham số luôn (trọng số và bias của các neuron). Hay nói cách khác, các lớp RELU/POOL thực thi một hàm cố định. Tham số nằm trong các lớp CONV/FC sẽ được trained theo gradient descent vì thế score cho mỗi class mà ConvNet tính toán được sẽ phải được khớp với nhãn khi training cho mỗi ảnh. 

Tóm tắt nhỏ:

- Một kiến trúc ConvNet ở dạng đơn giản nhất, là một dãy các  Layers thực hiện biến đổi ảnh đầu vào thành đầu ra (ví dụ, đầu ra chứa score cho mỗi class)
- Có vài loại Layers khác nhau (e.g. CONV/FC/RELU/POOL là phổ biến nhất)
- Mỗi lớp chấp nhận một đầu vào ở dạng 3 chiều và chuyển đổi nó sang một dạng 3 chiều ở đầu ra bằng một hàm khả vi 
- Mỗi lớp có thể có hoặc không có tham số (e.g. CONV/FC có, RELU/POOL thì không)
- Mỗi lớp có thể có hoặc không có các siêu tham số (e.g. CONV/FC/POOL có, RELU không)

<div class="fig figcenter fighighlight">
  <img src="/assets/cnn/convnet.jpeg" width="100%">
  <div class="figcaption">
    Một ví dụ về kiến trúc ConvNet. Khối ban đầu lưu các điểm ảnh ban đầu (phía trái) và khối cuối lưu score của mỗi class (phía phải). Mỗi phép kích hoạt sẽ theo các đường xử lý miêu tả trong cột tương ứng. Vì rất khó để mô phỏng khối 3 chiều, nên chúng tôi đặt mỗi lát của khối trên một hàng. Khối cuối cùng chứa score cho mỗi class, nhưng ở đây chúng tôi chỉ đưa ra 5 score cao nhất, và in ra nhãn tương ứng với chừng đó mà thôi. Danh sách đầy đủ <a href="http://cs231n.stanford.edu/">web-based demo</a> có ở đầu website. Kiến trúc sử dụng ở đây là tiny VGG Net, chúng ta sẽ bàn về nó ở dưới.
  </div>
</div>

Giờ chúng ta sẽ tìm hiểu các lớp và chi tiết các siêu tham số, các kết nối của mỗi lớp riêng biệt.

<a name='conv'></a>

#### Convolutional Layer

Lớp Conv là khối chính của Convolutional Network, thực hiện hầu hết các tính toán nặng. 

**Khái quát và trực quan mà coi như chưa biết đến những thứ liên quan đến bộ não.** Đầu tiên, hãy xem CONV tính toán cái gì khi chung ta chưa biết về khái niệm bộ não/neuron . Các tham số của lớp CONV chứa một tập các filter có khả năng học. Mỗi filter là một không gian hình học nhỏ (với chiều rộng và chiều cao), nhưng được áp dụng cho toàn bộ độ sâu của khối đầu vào. Ví dụ, một filter trong lớp đầu tiên của ConvNet có kích thước 5x5x3 chẳng hạn (i.e. 5 cho chiều rộng và chiều cao, và 3 vì ảnh có độ sâu 3, số kênh màu. Trong quá trình tính toán (fordward pass), chúng ta kéo (chính xác hơn, nhấc) filter để phủ hết chiều rộng, chiều cao của khối đầu vào và tính toán tích vô hướng giữa các phần tử trong filter và giá trị đầu vào ở bất cứ vị trí nào nó được kéo đến. Cứ làm như thế cho đến khi qua hết chiều rộng, chiều cao của khối đầu vào chúng ta sẽ có một bảng kết quả 2 chiều cho mọi ví trí trong không gian hình học. Trực quan hơn, mạng sẽ học được các filter mà kích hoạt khi chúng thấy có cạnh của hướng hoặc một đốm màu trong lớp đầu tiên, hoặc các mẫu cho thấy toàn bộ một tổ ong, hoặc cái trông giống giống bánh xe ở các lớp cao hơn. Giờ, chúng ta có toàn bộ tập filter rồi trong mỗi lớp CONV rồi (e.g. 12 filters), và mỗi cái trong chúng sẽ tạo ra một bảng kích hoạt riêng biệt. Chúng ta xếp các bảng này theo chiều độ sâu và tạo một khối đầu ra.

**Cách nhìn nhận từ khái niệm bộ não**. Nếu bạn là fan của các khái niệm bộ não/neuron, mỗi ô trong khối 3 chiều của đầu ra có thể được xem như đầu ra của của một neuron, cái mà chỉ xem xét một vùng nhỏ của đầu vào và chia sẻ tham số với tất cả các neuron ở bênh cạnh chúng (vì những giá trị này được tạo ra từ cùng 1 filter). Giờ chúng ta sẽ thảo luận chi tiết về kết nối giữa các neuron, sự sắp xếp trong không gian, và cách chúng chia sẻ tham số với nhau.

**Kết nối cục bộ.** Khi gặp những dữ liệu có số chiều lớn như hình ảnh, như đã thấy ở trên, sẽ không thực tế nếu kết nối mỗi neuron với tất cả các neuron của khối trước đó được. Thay vào đó, chúng ta sẽ kết nối mỗi neuron với một vùng của khối đầu vào mà thôi. Phạm vi không gian của kết nối loại này là một hyperparameter được gọi là **receptive field** của neuron đó (tương đương với kích thước filter ở đây). Phạm vi theo trục độ sâu luôn bằng với độ sâu của khối đầu vào. Một điểm quan trọng cần nhấn mạnh một lần nữa ở đây là sự bất đối xứng giữa không gian chiều (rộng và cao) với chiều sâu: Kết nối nối là cục bộ theo không gian (theo chiều rộng và chiều cao), nhưng luôn đầy đủ trên toàn chiều sâu của đầu vào.

*Ví dụ 1*. Lấy ví dụ, giả sử rằng khối đầu vào có kích thước là  [32x32x3], (e.g. ảnh RGB của CIFAR-10). Nếu receptive field (hay filter size) là 5x5, mỗi neuron trong Conv Layer sẽ có số trọng số cho một vùng [5x5x3] trong khối đầu vào tổng cộng 5\*5\*3 = 75 weights (và +1 bias parameter). Chú ý là phạm vi kết nối theo theo chiều sâu phải là 3, vì đây là độ sâu khối đầu vào.

*Example 2*. Giả sử khối đầu vào có kích thước [16x16x20]. Sau đó sử dụng một receptive field kích thước 3x3, thì mỗi neuron trong Conv Layer có tổng số kết nối là 3\*3\*20 = 180 đến khối đầu vào. Một lần nữa cần chú ý, kết nối kiểu này là cục bộ trong không gian (e.g. 3x3), nhưng đầy đủ theo chiều sâu (20).

<div class="fig figcenter fighighlight">
  <img src="/assets/cnn/depthcol.jpeg" width="40%">
  <img src="/assets/nn1/neuron_model.jpeg" width="40%" style="border-left: 1px solid black;">
  <div class="figcaption">
    <b>Left:</b> An example input volume in red (e.g. a 32x32x3 CIFAR-10 image), and an example volume of neurons in the first Convolutional layer. Each neuron in the convolutional layer is connected only to a local region in the input volume spatially, but to the full depth (i.e. all color channels). Note, there are multiple neurons (5 in this example) along the depth, all looking at the same region in the input - see discussion of depth columns in text below. <b>Right:</b> The neurons from the Neural Network chapter remain unchanged: They still compute a dot product of their weights with the input followed by a non-linearity, but their connectivity is now restricted to be local spatially.
  </div>
</div>

**Sắp xếp không gian**. Chúng ta đã giải thích về kết nối của mỗi neuron trong Conv Layer với khối đầu vào của nó, nhưng chúng ta chưa thảo luận xem có bao nhiêu neuron trong khối đầu ra hay chúng được sắp xếp như thế nào. Có 3 hyperparameters quyết định kích thước của khối đầu ra: **độ sâu, bước di chuyển** và **zero-padding**. Chúng ta sẽ theo luận nó ngay đây:

1. Đầu tiên, **độ sâu** của khối đầu ra là một hyperparameter: nó tương ứng với số filter mà chúng ta định sử dụng, mỗi cái sẽ quan sát một đặc điểm khác nhau ở đầu vào. Ví dụ, lớp Convolutional đầu tiên lấy dữ liệu ban đầu làm đầu vào, thì mỗi neurou dọc theo chiều sâu có thể phát hiện sự xuất hiện của các loại cạnh có hướng, hoặc đốm màu. Chúng ta sẽ coi tập các neuron cùng nhìn vào một vùng của đầu vào là **cột độ sâu** (một số người thích gọi là *fibre* hơn).
2. Thứ hai, chúng ta phải chỉ ra **bước di chuyển(stride)** lúc mà chúng ta kéo filter đi. Khi stride là 1 thì chúng ta sẽ di chuyển filter mỗi lần 1 pixel. Khi stride là 2 (hoặc ít phổ biến hơn là 3 hoặc hơn, được xem là hiếm trong thực tế) thì filter sẽ nhảy 2 pixels cho mỗi lần di chuyển sang bị trí tiếp theo. Khi stride càng lớn thì khối đầu ra có số chiều càng nhỏ lại.
3. Chúng ta sẽ sớm thấy rằng, thỉnh thoảng sẽ khá tiện khi pad (đưa thêm) vào biên dữ liệu vào một lượng hàng, cột zero. Kích thước của cái **zero-padding** này là một hyperparameter. Điểm rất hay của zero padding cho phép chúng ta quyết định kích thước không gian của khối đầu ra (hầu hết trường hợp chúng ta sẽ sớm thấy là chúng ta sẽ sử dụng nó để điều chỉnh chính xác kích thước của không gian đầu vào để ở đầu vào và đầu ra, giá trị chiều rộng , chiều cao sẽ bằng nhau).

Chúng ta sẽ tính toán khối đầu ra bằng hàm nhận các đầu vào là: kích thước khối đầu vào (\\(W\\)), kích thước receptive field của Conv Layer neurons (\\(F\\)), stride (\\(S\\)), số lượng zero-padding (\\(P\\)) trên biên. Bạn có thể tự chứng minh công thức tính số neuron sẽ có "fit" sẽ có dạng: \\((W - F + 2P)/S + 1\\). Ví dụ, đầu vào kích thước 7x7 và filter kích thước 3x3 filter, stride bằng 1, pad bằng 0, chúng ta sẽ có đầu ra kích thước 5x5. Với stride bằng 2, chúng ta sẽ có 3x3. Cùng xem thêm ví dụ bằng hình vẽ:

<div class="fig figcenter fighighlight">
  <img src="/assets/cnn/stride.jpeg">
  <div class="figcaption">
    Minh họa việc sắp xếp không gian. Trong ví dụ này, chỉ có một chiều không gian (x-axis), một neuron với receptive field size là F = 3, kích thước đầu vào W = 5, và zero padding P = 1. <b>Phía trái:</b> Các neuron được kéo trên đầu ra với stride S = 1, sẽ cho đầu ra kích thước (5 - 3 + 2)/1+1 = 5. <b>Phía phải:</b> Các neuron sử dụng stride S = 2, sẽ cho đầu ra kích thước (5 - 3 + 2)/2+1 = 3. Chú ý là S = 3 không sử dụng được bởi vì nó sẽ không khớp với các giá trị kích thước của khối. Về mặt biểu thức, điều này được đưa ra vì (5 - 3 + 2) = 4 không chia hết cho 3. 
    <br>Trọng số của neuron trong ví dụ này là [1,0,-1] (giá trị hiện ra bên phải), và bias của nó là zero. Trọng số này được chia sẻ giữa tất cả các neuron màu vàng (xem thêm phần chia sẻ tham số bên dưới).
  </div>
</div>

*Sử dụng zero-padding*. Trong ví dụ trên, ở phía trái, để ý rằng số chiều đầu vào là 5 và số chiều đầu ra: cũng là 5. Điều này xảy ra vì receptive fields đã là 3 và chúng ta đã sử dụng zero padding bằng 1. Nếu không sử dụng zero-padding, thì khối đầu ra sẽ có số chiều chỉ là 3, bởi vì đó sẽ tương ứng với số neuron "khớp" với đầu vào. Nói chung, thiết lập giá trị zero padding nên bằng \\(P = (F - 1)/2\\) khi strides \\(S = 1\\) sẽ đảm bảo rằng khối đầu vào và đầu ra có cùng kích thước không gian. Việc sử dụng zero-padding kiểu này thì rất phổ biến và chúng ta sẽ thảo luận lý do đầy đủ khi hiểu thêm về kiến trúc ConvNet.

*Ràng buộc vào strides*. Chú ý lần nữa rằng, các siêu tham số liên quan đến sắp xếp không gian có ràng buộc lẫn nhau. Ví dụ, khi đầu vào có kích thước \\(W = 10\\), không zero-padding \\(P = 0\\), và kích thước filter \\(F = 3\\), thì không thể sử dụng stride có \\(S = 2\\) được, vì \\((W - F + 2P)/S + 1 = (10 - 3 + 0) / 2 + 1 = 4.5\\), i.e. không nguyên, dẫn đến số neuron không "khớp" và phủ đối xứng đầu vào. Vì thế, bộ hyperparameters sẽ được coi là không hợp lệ, và một thư viện ConvNet có thể ném ra một ngoại lệ hoặc zero-pad phần còn lại để cho vừa, hoặc gọt đầu vào cho khớp, hoặc gì đó nữa. Bạn sẽ sớm thấy trong chương về kiến trúc ConvNet, việc điều chỉnh ConvNets để mọi kích thước có thể "chạy được" có thể hơi đau đầu đấy, trong khi sử dụng đúng zero-padding và một vài hướng dẫn thiết kế có thể nhẹ nhàng hơn nhiều.

*Ví dụ thật - Real-world example*. Kiến trúc mạng [Krizhevsky et al.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks), cái mà đã chiến thắng ImageNet challenge in 2012 sử dụng ảnh đầu vào có kích thước [227x227x3]. Trong lớp Convolutional đầu tiên, nó sử dụng receptive field kích thước \\(F = 11\\), stride \\(S = 4\\) và không zero-padding \\(P = 0\\). Vì (227 - 11)/4 + 1 = 55, và vì Conv layer đó có độ sâu \\(K = 96\\), nên khối đầu ra của Conv này sẽ có kích thước [55x55x96]. Mỗi neuron trong 55\*55\*96 neurons của khối đầu ra được kết nối đến một vùng nhỏ có kích thước [11x11x3] trên khối đầu vào. Hơn nữa, tất cả 96 neurons ở mỗi cột động sâu được kết nối đến cùng một vùng có kích thước [11x11x3] trên đầu vào, và tất nhiên với trọng số khác nhau. Có một chuyện khá vui, Nếu bạn đã đọc paper rồi thì bạn sẽ thắc mắc rằng kích thước ảnh được ghi là 224x224, nhưng như thế không đúng vì (224 - 11)/4 + 1 rõ ràng không thể là một số nguyên được. Điều này gây ra khó hiểu cho rất nhiều người khi nói về lịch sử của ConvNets và muốn biết chỗ đó là như nào. Suy đoán của tôi là Alex đã dùng zero-padding 3 điểm ảnh nhưng anh ấy không nhắc tới nó trong paper.

**Chia sẻ tham số - Parameter Sharing.** Mô hình chia sẻ tham số được sử dụng trong các lớp Convolutional để kiếm soát lượng tham số. Trong ví dụ thực tế ở trên, we see that there are 55\*55\*96 = 290,400 neurons in the first Conv Layer, and each has 11\*11\*3 = 363 weights and 1 bias. Together, this adds up to 290400 * 364 = 105,705,600 parameters on the first layer of the ConvNet alone. Clearly, this number is very high.

It turns out that we can dramatically reduce the number of parameters by making one reasonable assumption: That if one feature is useful to compute at some spatial position (x,y), then it should also be useful to compute at a different position (x2,y2). In other words, denoting a single 2-dimensional slice of depth as a **depth slice** (e.g. a volume of size [55x55x96] has 96 depth slices, each of size [55x55]), we are going to constrain the neurons in each depth slice to use the same weights and bias. With this parameter sharing scheme, the first Conv Layer in our example would now have only 96 unique set of weights (one for each depth slice), for a total of 96\*11\*11\*3 = 34,848 unique weights, or 34,944 parameters (+96 biases). Alternatively, all 55\*55 neurons in each depth slice will now be using the same parameters. In practice during backpropagation, every neuron in the volume will compute the gradient for its weights, but these gradients will be added up across each depth slice and only update a single set of weights per slice.

Notice that if all neurons in a single depth slice are using the same weight vector, then the forward pass of the CONV layer can in each depth slice be computed as a **convolution** of the neuron's weights with the input volume (Hence the name: Convolutional Layer). This is why it is common to refer to the sets of weights as a **filter** (or a **kernel**), that is convolved with the input.

<div class="fig figcenter fighighlight">
  <img src="/assets/cnn/weights.jpeg">
  <div class="figcaption">
    Example filters learned by Krizhevsky et al. Each of the 96 filters shown here is of size [11x11x3], and each one is shared by the 55*55 neurons in one depth slice. Notice that the parameter sharing assumption is relatively reasonable: If detecting a horizontal edge is important at some location in the image, it should intuitively be useful at some other location as well due to the translationally-invariant structure of images. There is therefore no need to relearn to detect a horizontal edge at every one of the 55*55 distinct locations in the Conv layer output volume.
  </div>
</div>

Note that sometimes the parameter sharing assumption may not make sense. This is especially the case when the input images to a ConvNet have some specific centered structure, where we should expect, for example, that completely different features should be learned on one side of the image than another. One practical example is when the input are faces that have been centered in the image. You might expect that different eye-specific or hair-specific features could (and should) be learned in different spatial locations. In that case it is common to relax the parameter sharing scheme, and instead simply call the layer a **Locally-Connected Layer**.

**Numpy examples.** To make the discussion above more concrete, lets express the same ideas but in code and with a specific example. Suppose that the input volume is a numpy array `X`. Then:

- A *depth column* (or a *fibre*) at position `(x,y)` would be the activations `X[x,y,:]`.
- A *depth slice*, or equivalently an *activation map* at depth `d` would be the activations `X[:,:,d]`. 

*Conv Layer Example*. Suppose that the input volume `X` has shape `X.shape: (11,11,4)`. Suppose further that we use no zero padding (\\(P = 0\\)), that the filter size is \\(F = 5\\), and that the stride is \\(S = 2\\). The output volume would therefore have spatial size (11-5)/2+1 = 4, giving a volume with width and height of 4. The activation map in the output volume (call it `V`), would then look as follows (only some of the elements are computed in this example):

- `V[0,0,0] = np.sum(X[:5,:5,:] * W0) + b0`
- `V[1,0,0] = np.sum(X[2:7,:5,:] * W0) + b0`
- `V[2,0,0] = np.sum(X[4:9,:5,:] * W0) + b0`
- `V[3,0,0] = np.sum(X[6:11,:5,:] * W0) + b0`

Remember that in numpy, the operation `*` above denotes elementwise multiplication between the arrays. Notice also that the weight vector `W0` is the weight vector of that neuron and `b0` is the bias. Here, `W0` is assumed to be of shape `W0.shape: (5,5,4)`, since the filter size is 5 and the depth of the input volume is 4. Notice that at each point, we are computing the dot product as seen before in ordinary neural networks. Also, we see that we are using the same weight and bias (due to parameter sharing), and where the dimensions along the width are increasing in steps of 2 (i.e. the stride). To construct a second activation map in the output volume, we would have:

- `V[0,0,1] = np.sum(X[:5,:5,:] * W1) + b1`
- `V[1,0,1] = np.sum(X[2:7,:5,:] * W1) + b1`
- `V[2,0,1] = np.sum(X[4:9,:5,:] * W1) + b1`
- `V[3,0,1] = np.sum(X[6:11,:5,:] * W1) + b1`
- `V[0,1,1] = np.sum(X[:5,2:7,:] * W1) + b1` (example of going along y)
- `V[2,3,1] = np.sum(X[4:9,6:11,:] * W1) + b1` (or along both)

where we see that we are indexing into the second depth dimension in `V` (at index 1) because we are computing the second activation map, and that a different set of parameters (`W1`) is now used. In the example above, we are for brevity leaving out some of the other operations the Conv Layer would perform to fill the other parts of the output array `V`. Additionally, recall that these activation maps are often followed elementwise through an activation function such as ReLU, but this is not shown here.

**Summary**. To summarize, the Conv Layer:

- Accepts a volume of size \\(W_1 \times H_1 \times D_1\\)
- Requires four hyperparameters: 
  - Number of filters \\(K\\), 
  - their spatial extent \\(F\\), 
  - the stride \\(S\\), 
  - the amount of zero padding \\(P\\).
- Produces a volume of size \\(W_2 \times H_2 \times D_2\\) where:
  - \\(W_2 = (W_1 - F + 2P)/S + 1\\)
  - \\(H_2 = (H_1 - F + 2P)/S + 1\\) (i.e. width and height are computed equally by symmetry)
  - \\(D_2 = K\\)
- With parameter sharing, it introduces \\(F \cdot F \cdot D_1\\) weights per filter, for a total of \\((F \cdot F \cdot D_1) \cdot K\\) weights and \\(K\\) biases.
- In the output volume, the \\(d\\)-th depth slice (of size \\(W_2 \times H_2\\)) is the result of performing a valid convolution of the \\(d\\)-th filter over the input volume with a stride of \\(S\\), and then offset by \\(d\\)-th bias.

A common setting of the hyperparameters is \\(F = 3, S = 1, P = 1\\). However, there are common conventions and rules of thumb that motivate these hyperparameters. See the [ConvNet architectures](#architectures) section below.

**Convolution Demo**. Below is a running demo of a CONV layer. Since 3D volumes are hard to visualize, all the volumes (the input volume (in blue), the weight volumes (in red), the output volume (in green)) are visualized with each depth slice stacked in rows. The input volume is of size \\(W_1 = 5, H_1 = 5, D_1 = 3\\), and the CONV layer parameters are \\(K = 2, F = 3, S = 2, P = 1\\). That is, we have two filters of size \\(3 \times 3\\), and they are applied with a stride of 2. Therefore, the output volume size has spatial size (5 - 3 + 2)/2 + 1 = 3. Moreover, notice that a padding of \\(P = 1\\) is applied to the input volume, making the outer border of the input volume zero. The visualization below iterates over the output activations (green), and shows that each element is computed by elementwise multiplying the highlighted input (blue) with the filter (red), summing it up, and then offsetting the result by the bias.

<div class="fig figcenter fighighlight">
  <iframe src="/assets/conv-demo/index.html" width="100%" height="700px;" style="border:none;"></iframe>
  <div class="figcaption"></div>
</div>

**Implementation as Matrix Multiplication**. Note that the convolution operation essentially performs dot products between the filters and local regions of the input. A common implementation pattern of the CONV layer is to take advantage of this fact and formulate the forward pass of a convolutional layer as one big matrix multiply as follows:

1. The local regions in the input image are stretched out into columns in an operation commonly called **im2col**. For example, if the input is [227x227x3] and it is to be convolved with 11x11x3 filters at stride 4, then we would take [11x11x3] blocks of pixels in the input and stretch each block into a column vector of size 11\*11\*3 = 363. Iterating this process in the input at stride of 4 gives (227-11)/4+1 = 55 locations along both width and height, leading to an output matrix `X_col` of *im2col* of size [363 x 3025], where every column is a stretched out receptive field and there are 55*55 = 3025 of them in total. Note that since the receptive fields overlap, every number in the input volume may be duplicated in multiple distinct columns.
2. The weights of the CONV layer are similarly stretched out into rows. For example, if there are 96 filters of size [11x11x3] this would give a matrix `W_row` of size [96 x 363].
3. The result of a convolution is now equivalent to performing one large matrix multiply `np.dot(W_row, X_col)`, which evaluates the dot product between every filter and every receptive field location. In our example, the output of this operation would be [96 x 3025], giving the output of the dot product of each filter at each location. 
4. The result must finally be reshaped back to its proper output dimension [55x55x96].

This approach has the downside that it can use a lot of memory, since some values in the input volume are replicated multiple times in `X_col`. However, the benefit is that there are many very efficient implementations of Matrix Multiplication that we can take advantage of (for example, in the commonly used [BLAS](http://www.netlib.org/blas/) API). Moreover, the same *im2col* idea can be reused to perform the pooling operation, which we discuss next.

**Backpropagation.** The backward pass for a convolution operation (for both the data and the weights) is also a convolution (but with spatially-flipped filters). This is easy to derive in the 1-dimensional case with a toy example (not expanded on for now).

**1x1 convolution**. As an aside, several papers use 1x1 convolutions, as first investigated by [Network in Network](http://arxiv.org/abs/1312.4400). Some people are at first confused to see 1x1 convolutions especially when they come from signal processing background. Normally signals are 2-dimensional so 1x1 convolutions do not make sense (it's just pointwise scaling). However, in ConvNets this is not the case because one must remember that we operate over 3-dimensional volumes, and that the filters always extend through the full depth of the input volume. For example, if the input is [32x32x3] then doing 1x1 convolutions would effectively be doing 3-dimensional dot products (since the input depth is 3 channels).

**Dilated convolutions.** A recent development (e.g. see [paper by Fisher Yu and Vladlen Koltun](https://arxiv.org/abs/1511.07122)) is to introduce one more hyperparameter to the CONV layer called the *dilation*. So far we've only discussed CONV filters that are contiguous. However, it's possible to have filters that have spaces between each cell, called dilation. As an example, in one dimension a filter `w` of size 3 would compute over input `x` the following: `w[0]*x[0] + w[1]*x[1] + w[2]*x[2]`. This is dilation of 0. For dilation 1 the filter would instead compute `w[0]*x[0] + w[1]*x[2] + w[2]*x[4]`; In other words there is a gap of 1 between the applications. This can be very useful in some settings to use in conjunction with 0-dilated filters because it allows you to merge spatial information across the inputs much more agressively with fewer layers. For example, if you stack two 3x3 CONV layers on top of each other then you can convince yourself that the neurons on the 2nd layer are a function of a 5x5 patch of the input (we would say that the *effective receptive field* of these neurons is 5x5). If we use dilated convolutions then this effective receptive field would grow much quicker.

<a name='pool'></a>

#### Pooling Layer

It is common to periodically insert a Pooling layer in-between successive Conv layers in a ConvNet architecture. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting. The Pooling Layer operates independently on every depth slice of the input and resizes it spatially, using the MAX operation. The most common form is a pooling layer with filters of size 2x2 applied with a stride of 2 downsamples every depth slice in the input by 2 along both width and height, discarding 75% of the activations. Every MAX operation would in this case be taking a max over 4 numbers (little 2x2 region in some depth slice). The depth dimension remains unchanged. More generally, the pooling layer:

- Accepts a volume of size \\(W_1 \times H_1 \times D_1\\)
- Requires two hyperparameters: 
  - their spatial extent \\(F\\), 
  - the stride \\(S\\), 
- Produces a volume of size \\(W_2 \times H_2 \times D_2\\) where:
  - \\(W_2 = (W_1 - F)/S + 1\\)
  - \\(H_2 = (H_1 - F)/S + 1\\)
  - \\(D_2 = D_1\\)
- Introduces zero parameters since it computes a fixed function of the input
- Note that it is not common to use zero-padding for Pooling layers

It is worth noting that there are only two commonly seen variations of the max pooling layer found in practice: A pooling layer with \\(F = 3, S = 2\\) (also called overlapping pooling), and more commonly \\(F = 2, S = 2\\). Pooling sizes with larger receptive fields are too destructive.

**General pooling**. In addition to max pooling, the pooling units can also perform other functions, such as *average pooling* or even *L2-norm pooling*. Average pooling was often used historically but has recently fallen out of favor compared to the max pooling operation, which has been shown to work better in practice.

<div class="fig figcenter fighighlight">
  <img src="/assets/cnn/pool.jpeg" width="36%">
  <img src="/assets/cnn/maxpool.jpeg" width="59%" style="border-left: 1px solid black;">
  <div class="figcaption">
    Pooling layer downsamples the volume spatially, independently in each depth slice of the input volume. <b>Left:</b> In this example, the input volume of size [224x224x64] is pooled with filter size 2, stride 2 into output volume of size [112x112x64]. Notice that the volume depth is preserved. <b>Right:</b> The most common downsampling operation is max, giving rise to <b>max pooling</b>, here shown with a stride of 2. That is, each max is taken over 4 numbers (little 2x2 square).
  </div>
</div>

**Backpropagation**. Recall from the backpropagation chapter that the backward pass for a max(x, y) operation has a simple interpretation as only routing the gradient to the input that had the highest value in the forward pass. Hence, during the forward pass of a pooling layer it is common to keep track of the index of the max activation (sometimes also called *the switches*) so that gradient routing is efficient during backpropagation.

**Getting rid of pooling**. Many people dislike the pooling operation and think that we can get away without it. For example, [Striving for Simplicity: The All Convolutional Net](http://arxiv.org/abs/1412.6806) proposes to discard the pooling layer in favor of architecture that only consists of repeated CONV layers. To reduce the size of the representation they suggest using larger stride in CONV layer once in a while. Discarding pooling layers has also been found to be important in training good generative models, such as variational autoencoders (VAEs) or generative adversarial networks (GANs). It seems likely that future architectures will feature very few to no pooling layers.

<a name='norm'></a>

#### Normalization Layer

Many types of normalization layers have been proposed for use in ConvNet architectures, sometimes with the intentions of implementing inhibition schemes observed in the biological brain. However, these layers have since fallen out of favor because in practice their contribution has been shown to be minimal, if any. For various types of normalizations, see the discussion in Alex Krizhevsky's [cuda-convnet library API](http://code.google.com/p/cuda-convnet/wiki/LayerParams#Local_response_normalization_layer_(same_map)).

<a name='fc'></a>

#### Fully-connected layer

Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. Their activations can hence be computed with a matrix multiplication followed by a bias offset. See the *Neural Network* section of the notes for more information.

<a name='convert'></a>

#### Converting FC layers to CONV layers 

It is worth noting that the only difference between FC and CONV layers is that the neurons in the CONV layer are connected only to a local region in the input, and that many of the neurons in a CONV volume share parameters. However, the neurons in both layers still compute dot products, so their functional form is identical. Therefore, it turns out that it's possible to convert between FC and CONV layers:

- For any CONV layer there is an FC layer that implements the same forward function. The weight matrix would be a large matrix that is mostly zero except for at certain blocks (due to local connectivity) where the weights in many of the blocks are equal (due to parameter sharing).
- Conversely, any FC layer can be converted to a CONV layer. For example, an FC layer with \\(K = 4096\\) that is looking at some input volume of size \\(7 \times 7 \times 512\\) can be equivalently expressed as a CONV layer with \\(F = 7, P = 0, S = 1, K = 4096\\). In other words, we are setting the filter size to be exactly the size of the input volume, and hence the output will simply be \\(1 \times 1 \times 4096\\) since only a single depth column "fits" across the input volume, giving identical result as the initial FC layer.

**FC->CONV conversion**. Of these two conversions, the ability to convert an FC layer to a CONV layer is particularly useful in practice. Consider a ConvNet architecture that takes a 224x224x3 image, and then uses a series of CONV layers and POOL layers to reduce the image to an activations volume of size 7x7x512 (in an *AlexNet* architecture that we'll see later, this is done by use of 5 pooling layers that downsample the input spatially by a factor of two each time, making the final spatial size 224/2/2/2/2/2 = 7). From there, an AlexNet uses two FC layers of size 4096 and finally the last FC layers with 1000 neurons that compute the class scores. We can convert each of these three FC layers to CONV layers as described above:

- Replace the first FC layer that looks at [7x7x512] volume with a CONV layer that uses filter size \\(F = 7\\), giving output volume [1x1x4096].
- Replace the second FC layer with a CONV layer that uses filter size \\(F = 1\\), giving output volume [1x1x4096]
- Replace the last FC layer similarly, with \\(F=1\\), giving final output [1x1x1000]

Each of these conversions could in practice involve manipulating (e.g. reshaping) the weight matrix \\(W\\) in each FC layer into CONV layer filters. It turns out that this conversion allows us to "slide" the original ConvNet very efficiently across many spatial positions in a larger image, in a single forward pass. 

For example, if 224x224 image gives a volume of size [7x7x512] - i.e. a reduction by 32, then forwarding an image of size 384x384 through the converted architecture would give the equivalent volume in size [12x12x512], since 384/32 = 12. Following through with the next 3 CONV layers that we just converted from FC layers would now give the final volume of size [6x6x1000], since (12 - 7)/1 + 1 = 6. Note that instead of a single vector of class scores of size [1x1x1000], we're now getting an entire 6x6 array of class scores across the 384x384 image.

> Evaluating the original ConvNet (with FC layers) independently across 224x224 crops of the 384x384 image in strides of 32 pixels gives an identical result to forwarding the converted ConvNet one time.

Naturally, forwarding the converted ConvNet a single time is much more efficient than iterating the original ConvNet over all those 36 locations, since the 36 evaluations share computation. This trick is often used in practice to get better performance, where for example, it is common to resize an image to make it bigger, use a converted ConvNet to evaluate the class scores at many spatial positions and then average the class scores.

Lastly, what if we wanted to efficiently apply the original ConvNet over the image but at a stride smaller than 32 pixels? We could achieve this with multiple forward passes. For example, note that if we wanted to use a stride of 16 pixels we could do so by combining the volumes received by forwarding the converted ConvNet twice: First over the original image and second over the image but with the image shifted spatially by 16 pixels along both width and height.

- An IPython Notebook on [Net Surgery](https://github.com/BVLC/caffe/blob/master/examples/net_surgery.ipynb) shows how to perform the conversion in practice, in code (using Caffe)

<a name='architectures'></a>

### ConvNet Architectures

We have seen that Convolutional Networks are commonly made up of only three layer types: CONV, POOL (we assume Max pool unless stated otherwise) and FC (short for fully-connected). We will also explicitly write the RELU activation function as a layer, which applies elementwise non-linearity. In this section we discuss how these are commonly stacked together to form entire ConvNets. 

<a name='layerpat'></a>

#### Layer Patterns
The most common form of a ConvNet architecture stacks a few CONV-RELU layers, follows them with POOL layers, and repeats this pattern until the image has been merged spatially to a small size. At some point, it is common to transition to fully-connected layers. The last fully-connected layer holds the output, such as the class scores. In other words, the most common ConvNet architecture follows the pattern:

`INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC`

where the `*` indicates repetition, and the `POOL?` indicates an optional pooling layer. Moreover, `N >= 0` (and usually `N <= 3`), `M >= 0`, `K >= 0` (and usually `K < 3`). For example, here are some common ConvNet architectures you may see that follow this pattern:

- `INPUT -> FC`, implements a linear classifier. Here `N = M = K = 0`.
- `INPUT -> CONV -> RELU -> FC`
- `INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC`. Here we see that there is a single CONV layer between every POOL layer.
- `INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC` Here we see two CONV layers stacked before every POOL layer. This is generally a good idea for larger and deeper networks, because multiple stacked CONV layers can develop more complex features of the input volume before the destructive pooling operation.

*Prefer a stack of small filter CONV to one large receptive field CONV layer*. Suppose that you stack three 3x3 CONV layers on top of each other (with non-linearities in between, of course). In this arrangement, each neuron on the first CONV layer has a 3x3 view of the input volume. A neuron on the second CONV layer has a 3x3 view of the first CONV layer, and hence by extension a 5x5 view of the input volume. Similarly, a neuron on the third CONV layer has a 3x3 view of the 2nd CONV layer, and hence a 7x7 view of the input volume. Suppose that instead of these three layers of 3x3 CONV, we only wanted to use a single CONV layer with 7x7 receptive fields. These neurons would have a receptive field size of the input volume that is identical in spatial extent (7x7), but with several disadvantages. First, the neurons would be computing a linear function over the input, while the three stacks of CONV layers contain non-linearities that make their features more expressive. Second, if we suppose that all the volumes have \\(C\\) channels, then it can be seen that the single 7x7 CONV layer would contain \\(C \times (7 \times 7 \times C) = 49 C^2\\) parameters, while the three 3x3 CONV layers would only contain \\(3 \times (C \times (3 \times 3 \times C)) = 27 C^2\\) parameters. Intuitively, stacking CONV layers with tiny filters as opposed to having one CONV layer with big filters allows us to express more powerful features of the input, and with fewer parameters. As a practical disadvantage, we might need more memory to hold all the intermediate CONV layer results if we plan to do backpropagation.

**Recent departures.** It should be noted that the conventional paradigm of a linear list of layers has recently been challenged, in Google's Inception architectures and also in current (state of the art) Residual Networks from Microsoft Research Asia. Both of these (see details below in case studies section) feature more intricate and different connectivity structures.

**In practice: use whatever works best on ImageNet**. If you're feeling a bit of a fatigue in thinking about the architectural decisions, you'll be pleased to know that in 90% or more of applications you should not have to worry about these. I like to summarize this point as "*don't be a hero*": Instead of rolling your own architecture for a problem, you should look at whatever architecture currently works best on ImageNet, download a pretrained model and finetune it on your data. You should rarely ever have to train a ConvNet from scratch or design one from scratch. I also made this point at the [Deep Learning school](https://www.youtube.com/watch?v=u6aEYuemt0M).

<a name='layersizepat'></a>

#### Layer Sizing Patterns

Until now we've omitted mentions of common hyperparameters used in each of the layers in a ConvNet. We will first state the common rules of thumb for sizing the architectures and then follow the rules with a discussion of the notation:

The **input layer** (that contains the image) should be divisible by 2 many times. Common numbers include 32 (e.g. CIFAR-10), 64, 96 (e.g. STL-10), or 224 (e.g. common ImageNet ConvNets), 384, and 512.

The **conv layers** should be using small filters (e.g. 3x3 or at most 5x5), using a stride of \\(S = 1\\), and crucially, padding the input volume with zeros in such way that the conv layer does not alter the spatial dimensions of the input. That is, when \\(F = 3\\), then using \\(P = 1\\) will retain the original size of the input. When \\(F = 5\\), \\(P = 2\\). For a general \\(F\\), it can be seen that \\(P = (F - 1) / 2\\) preserves the input size. If you must use bigger filter sizes (such as 7x7 or so), it is only common to see this on the very first conv layer that is looking at the input image.

The **pool layers** are in charge of downsampling the spatial dimensions of the input. The most common setting is to use max-pooling with 2x2 receptive fields (i.e. \\(F = 2\\)), and with a stride of 2 (i.e. \\(S = 2\\)). Note that this discards exactly 75% of the activations in an input volume (due to downsampling by 2 in both width and height). Another slightly less common setting is to use 3x3 receptive fields with a stride of 2, but this makes. It is very uncommon to see receptive field sizes for max pooling that are larger than 3 because the pooling is then too lossy and aggressive. This usually leads to worse performance.

*Reducing sizing headaches.* The scheme presented above is pleasing because all the CONV layers preserve the spatial size of their input, while the POOL layers alone are in charge of down-sampling the volumes spatially. In an alternative scheme where we use strides greater than 1 or don't zero-pad the input in CONV layers, we would have to very carefully keep track of the input volumes throughout the CNN architecture and make sure that all strides and filters "work out", and that the ConvNet architecture is nicely and symmetrically wired.

*Why use stride of 1 in CONV?* Smaller strides work better in practice. Additionally, as already mentioned stride 1 allows us to leave all spatial down-sampling to the POOL layers, with the CONV layers only transforming the input volume depth-wise.

*Why use padding?* In addition to the aforementioned benefit of keeping the spatial sizes constant after CONV, doing this actually improves performance. If the CONV layers were to not zero-pad the inputs and only perform valid convolutions, then the size of the volumes would reduce by a small amount after each CONV, and the information at the borders would be "washed away" too quickly.

*Compromising based on memory constraints.* In some cases (especially early in the ConvNet architectures), the amount of memory can build up very quickly with the rules of thumb presented above. For example, filtering a 224x224x3 image with three 3x3 CONV layers with 64 filters each and padding 1 would create three activation volumes of size [224x224x64]. This amounts to a total of about 10 million activations, or 72MB of memory (per image, for both activations and gradients). Since GPUs are often bottlenecked by memory, it may be necessary to compromise. In practice, people prefer to make the compromise at only the first CONV layer of the network. For example, one compromise might be to use a first CONV layer with filter sizes of 7x7 and stride of 2 (as seen in a ZF net). As another example, an AlexNet uses filter sizes of 11x11 and stride of 4.

<a name='case'></a>

#### Case studies

There are several architectures in the field of Convolutional Networks that have a name. The most common are:

- **LeNet**. The first successful applications of Convolutional Networks were developed by Yann LeCun in 1990's. Of these, the best known is the [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) architecture that was used to read zip codes, digits, etc.
- **AlexNet**. The first work that popularized Convolutional Networks in Computer Vision was the [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks), developed by Alex Krizhevsky, Ilya Sutskever and Geoff Hinton. The AlexNet was submitted to the [ImageNet ILSVRC challenge](http://www.image-net.org/challenges/LSVRC/2014/) in 2012 and significantly outperformed the second runner-up (top 5 error of 16% compared to runner-up with 26% error). The Network had a very similar architecture to LeNet, but was deeper, bigger, and featured Convolutional Layers stacked on top of each other (previously it was common to only have a single CONV layer always immediately followed by a POOL layer).
- **ZF Net**. The ILSVRC 2013 winner was a Convolutional Network from Matthew Zeiler and Rob Fergus. It became known as the [ZFNet](http://arxiv.org/abs/1311.2901) (short for Zeiler & Fergus Net). It was an improvement on AlexNet by tweaking the architecture hyperparameters, in particular by expanding the size of the middle convolutional layers and making the stride and filter size on the first layer smaller.
- **GoogLeNet**. The ILSVRC 2014 winner was a Convolutional Network from [Szegedy et al.](http://arxiv.org/abs/1409.4842) from Google. Its main contribution was the development of an *Inception Module* that dramatically reduced the number of parameters in the network (4M, compared to AlexNet with 60M). Additionally, this paper uses Average Pooling instead of Fully Connected layers at the top of the ConvNet, eliminating a large amount of parameters that do not seem to matter much. There are also several followup versions to the GoogLeNet, most recently [Inception-v4](http://arxiv.org/abs/1602.07261).
- **VGGNet**. The runner-up in ILSVRC 2014 was the network from Karen Simonyan and Andrew Zisserman that became known as the [VGGNet](http://www.robots.ox.ac.uk/~vgg/research/very_deep/). Its main contribution was in showing that the depth of the network is a critical component for good performance. Their final best network contains 16 CONV/FC layers and, appealingly, features an extremely homogeneous architecture that only performs 3x3 convolutions and 2x2 pooling from the beginning to the end. Their [pretrained model](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) is available for plug and play use in Caffe. A downside of the VGGNet is that it is more expensive to evaluate and uses a lot more memory and parameters (140M). Most of these parameters are in the first fully connected layer, and it was since found that these FC layers can be removed with no performance downgrade, significantly reducing the number of necessary parameters.
- **ResNet**. [Residual Network](http://arxiv.org/abs/1512.03385) developed by Kaiming He et al. was the winner of ILSVRC 2015. It features special *skip connections* and a heavy use of [batch normalization](http://arxiv.org/abs/1502.03167). The architecture is also missing fully connected layers at the end of the network. The reader is also referred to Kaiming's presentation ([video](https://www.youtube.com/watch?v=1PGLj-uKT1w), [slides](http://research.microsoft.com/en-us/um/people/kahe/ilsvrc15/ilsvrc2015_deep_residual_learning_kaiminghe.pdf)), and some [recent experiments](https://github.com/gcr/torch-residual-networks) that reproduce these networks in Torch. ResNets are currently by far state of the art Convolutional Neural Network models and are the default choice for using ConvNets in practice (as of May 10, 2016). In particular, also see more recent developments that tweak the original architecture from [Kaiming He et al. Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (published March 2016).

**VGGNet in detail**.
Lets break down the [VGGNet](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) in more detail as a case study. The whole VGGNet is composed of CONV layers that perform 3x3 convolutions with stride 1 and pad 1, and of POOL layers that perform 2x2 max pooling with stride 2 (and no padding). We can write out the size of the representation at each step of the processing and keep track of both the representation size and the total number of weights:

```
INPUT: [224x224x3]        memory:  224*224*3=150K   weights: 0
CONV3-64: [224x224x64]  memory:  224*224*64=3.2M   weights: (3*3*3)*64 = 1,728
CONV3-64: [224x224x64]  memory:  224*224*64=3.2M   weights: (3*3*64)*64 = 36,864
POOL2: [112x112x64]  memory:  112*112*64=800K   weights: 0
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M   weights: (3*3*64)*128 = 73,728
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M   weights: (3*3*128)*128 = 147,456
POOL2: [56x56x128]  memory:  56*56*128=400K   weights: 0
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*128)*256 = 294,912
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
POOL2: [28x28x256]  memory:  28*28*256=200K   weights: 0
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*256)*512 = 1,179,648
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
POOL2: [14x14x512]  memory:  14*14*512=100K   weights: 0
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
POOL2: [7x7x512]  memory:  7*7*512=25K  weights: 0
FC: [1x1x4096]  memory:  4096  weights: 7*7*512*4096 = 102,760,448
FC: [1x1x4096]  memory:  4096  weights: 4096*4096 = 16,777,216
FC: [1x1x1000]  memory:  1000 weights: 4096*1000 = 4,096,000

TOTAL memory: 24M * 4 bytes ~= 93MB / image (only forward! ~*2 for bwd)
TOTAL params: 138M parameters
```

As is common with Convolutional Networks, notice that most of the memory (and also compute time) is used in the early CONV layers, and that most of the parameters are in the last FC layers. In this particular case, the first FC layer contains 100M weights, out of a total of 140M.


<a name='comp'></a>

#### Computational Considerations

The largest bottleneck to be aware of when constructing ConvNet architectures is the memory bottleneck. Many modern GPUs have a limit of 3/4/6GB memory, with the best GPUs having about 12GB of memory. There are three major sources of memory to keep track of:

- From the intermediate volume sizes: These are the raw number of **activations** at every layer of the ConvNet, and also their gradients (of equal size). Usually, most of the activations are on the earlier layers of a ConvNet (i.e. first Conv Layers). These are kept around because they are needed for backpropagation, but a clever implementation that runs a ConvNet only at test time could in principle reduce this by a huge amount, by only storing the current activations at any layer and discarding the previous activations on layers below.
- From the parameter sizes: These are the numbers that hold the network **parameters**, their gradients during backpropagation, and commonly also a step cache if the optimization is using momentum, Adagrad, or RMSProp. Therefore, the memory to store the parameter vector alone must usually be multiplied by a factor of at least 3 or so.
- Every ConvNet implementation has to maintain **miscellaneous** memory, such as the image data batches, perhaps their augmented versions, etc.

Once you have a rough estimate of the total number of values (for activations, gradients, and misc), the number should be converted to size in GB. Take the number of values, multiply by 4 to get the raw number of bytes (since every floating point is 4 bytes, or maybe by 8 for double precision), and then divide by 1024 multiple times to get the amount of memory in KB, MB, and finally GB. If your network doesn't fit, a common heuristic to "make it fit" is to decrease the batch size, since most of the memory is usually consumed by the activations.


<a name='add'></a>

### Additional Resources

Additional resources related to implementation:

- [Soumith benchmarks for CONV performance](https://github.com/soumith/convnet-benchmarks)
- [ConvNetJS CIFAR-10 demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html) allows you to play with ConvNet architectures and see the results and computations in real time, in the browser.
- [Caffe](http://caffe.berkeleyvision.org/), one of the popular ConvNet libraries.
- [State of the art ResNets in Torch7](http://torch.ch/blog/2016/02/04/resnets.html)
