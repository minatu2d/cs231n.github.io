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

**Chia sẻ tham số - Parameter Sharing.** Mô hình chia sẻ tham số được sử dụng trong các lớp Convolutional để kiếm soát lượng tham số. Trong ví dụ thực tế ở trên, chúng ta thấy có đến 55\*55\*96 = 290,400 neurons trong lớp Conv đầu tiên, mỗi neuron này có 11\*11\*3 = 363 trọng số và 1 bias. Cộng hết vào, chúng ta sẽ có 290400 * 364 = 105,705,600 tham số chỉ trong lớp đầu tiên của ConvNet. Rõ ràng, con số này quá lớn.

Thực tế chứng minh rằng, chúng ta có thể giảm đáng kể lượng tham số bằng cách đưa ra một giả định hợp lý rằng: đó là, nếu một đặc tính - feature hữu dụng để tính toán ở một vài vị trí (x,y), thì nó cũng hữu dụng để tính toán ở một vị trí (x2,y2) khác. Nói cách khác, biểu diễn độ dịch chuyển một mảng 2 chiều theo là **depth slice (bước theo sâu)** (ví dụ: một khối có kích thước [55x55x96] có 96 bước, mỗi cái có kích thước [55x55]), chúng ta sẽ ràng buộc là các neuron ở cùng độ sâu sử dụng cùng một tập trọng số và bias. Với mô hình chia sẻ tham số này, lớp Conv đầu tiên trong ví dụ của chúng ta chỉ có 96 tập trọng số riêng biệt mà thôi (mỗi độ sâu có một cái), vậy tổng sẽ là 96\*11\*11\*3 = 34,848 trọng số riêng biệt, hay 34,944 tham số (+96 biases). Tức là, tất cả 55\*55 neurons ở cùng một độ sâu thì sử dụng chung tham số. Trong thực tế, khi phản hồi ngược, mỗi neuron trong khối sẽ tính toán sẽ tính gradient ứng với trọng số của nó, nhưng những gradients này sẽ được cộng trên toàn độ sâu đó và chỉ cập nhật một tập trọng số trên mỗi độ sâu thôi.

Chú ý là, nếu tất cả các neuron trong cùng một độ sâu sử dụng cùng vector trọng số, thì bước forward pass của lớp CONV trong mỗi độ sâu được tính toán sẽ giống như việc xoắn **convolution** các trọng số của neuron với khối đầu vào (vậy mới có cái tên: Convolutional Layer). Đây cũng là lý do tạo sao người ta gọi tập các trọng số là bộ lọc **filter** (hoặc nhân **kernel**), cái được cuốn với đầu vào.

<div class="fig figcenter fighighlight">
  <img src="/assets/cnn/weights.jpeg">
  <div class="figcaption">
    Filters ví dụ được học trong Krizhevsky et al. Mỗi tập trong 96 filters có kích thước là [11x11x3], và được chia sẻ bởi 55*55 neurons ở cùng một độ sâu. Chú ý rằng giả định nền tảng cho việc chia sẻ tham số là tương đối hợp lý: Nếu tính toán để phát hiện cạnh ngang ở một số vị trí trên ảnh, về mặt trực giác thì tính toán đó cũng có thể hữu ích ở một số vị trí khác do cấu trúc chuyển dịch bất biến của ảnh. Chính vì lé đó, sẽ không cần học lại để phát hiện một cạnh ngang ở mỗi vị trí trong tất cả các vị trí của 55x55 trong khối đầu ra của Conv nữa.
  </div>
</div>

Chú ý rằng, thỉnh thoảng giả định chia sẻ tham số này cũng không hiệu quả. Đây là trường hợp đặc biệt khi ảnh đầu vào cho ConvNet có cấu trúc căn giữa nhất định, khi đó cái chúng ta mong muốn, các đặc tính - features học được từ một phía của ảnh có thể hoàn toàn khác với chiều khác. Một ví dụ thực tế là khi đầu vào là nhiều khuôn mặt nằm ở giữa ảnh. Bạn có thể mong muốn rằng đặc tính cụ thể của mắt, tóc có thể được học từ các vị trí không gian khác nhau. Trong trường hợp đó, người ta thường nới lỏng mô hình chia sẻ tham số, và đơn giản gọi một lớp đó là **Locally-Connected Layer**.

**Ví dụ bằng Numpy.** Để biến những thảo luận ở trên cụ thể hơn, cùng ý tưởng như trên, hãy cùng nhau biểu diễn trong code qua các ví dụ cụ thể. Giả định khối đầu vào là một mảng numy `X`. Thì:

- Cột theo độ sâu - *depth column* (hay gọi *fibre*) ở vị trí `(x,y)` sẽ là `X[x,y,:]`.
- Bảng tương ứng với độ sâu - *depth slice*, hoặc bảng kích hoạt -  *activation map* ở độ sâu `d` sẽ là `X[:,:,d]`. 

*Conv Layer Example*. Giả sử, khối đầu vào `X` có hình dạng `X.shape: (11,11,4)`. Giả định thêm nữa, chúng ta không sử dụng zerp-padding (\\(P = 0\\)), rồi thì kích thước filter \\(F = 5\\), bước kéo filter \\(S = 2\\). Kích thước khối đầu ra sẽ được tính bằng (11-5)/2+1 = 4, một khối có kích thước hàng, cột là 4. Bảng kích hoạt ở khối đầu ra (gọi nó là `V`), trong sẽ giống như sau (chỉ một vài phần tử được tính trong ví dụ này):

- `V[0,0,0] = np.sum(X[:5,:5,:] * W0) + b0`
- `V[1,0,0] = np.sum(X[2:7,:5,:] * W0) + b0`
- `V[2,0,0] = np.sum(X[4:9,:5,:] * W0) + b0`
- `V[3,0,0] = np.sum(X[6:11,:5,:] * W0) + b0`

Nhớ rằng trong numpy, phép `*` biểu diễn phép toán nhân từng phần tử trong mảng. Và chú ý đến vector `W0` là vector trọng số của neuron và `b0` là bias. Ở đây, `W0` được giả định có dạng hình học `W0.shape: (5,5,4)`, vì kích thước filter là 5 và độ sâu của khối đầu vào là 4. Chú ý là, ở mỗi điểm, chúng ta sẽ thực hiện phép nhân vô hướng như đã thấy ở mạng neural thông thương. Cũng ở đây, chúng ta thấy rằng chúng ta đang sử dụng cùng weight and bias (do parameter sharing), các mảng được dịch dần theo chiều rộng với bước kéo là 2 (chính là stride). Để tính bảng activation map thứ hai trong khối đầu ra, chúng ta sẽ làm như sau:

- `V[0,0,1] = np.sum(X[:5,:5,:] * W1) + b1`
- `V[1,0,1] = np.sum(X[2:7,:5,:] * W1) + b1`
- `V[2,0,1] = np.sum(X[4:9,:5,:] * W1) + b1`
- `V[3,0,1] = np.sum(X[6:11,:5,:] * W1) + b1`
- `V[0,1,1] = np.sum(X[:5,2:7,:] * W1) + b1` (example of going along y)
- `V[2,3,1] = np.sum(X[4:9,6:11,:] * W1) + b1` (or along both)

Ở chúng ta thấy rằng chúng ta đang thao tác với độ sâu thứ hai của `V` (tức là index 1) vì chúng ta đang tính bảng activation map thứ 2, và một tập tham số khác (`W1`) được sử dụng. Ở ví dụ trên, chúng ta đã ngầm định bỏ qua một số thao tác khác cho lớp Conv Layer để điền đầy đủ các giá trị của `V`. Thêm vào đó, nhớ lại một chút rằng, các bảng activation maps thường được đưa qua các hàm activation như ReLU, nhưng cái đó không được nói ở đây.

**Tóm tắt**. Đến đây, chúng ta có Conv Layer:

- Nhận vào một khối có kích thước \\(W_1 \times H_1 \times D_1\\)
- Cần 4 siêu tham số: 
  - Số filters \\(K\\), 
  - Phạm vi về số chiều \\(F\\), 
  - Bước khi kéo qua khối đầu vào \\(S\\), 
  - Số lượng zero-padding \\(P\\).
- Tạo ra một khối có kích thước \\(W_2 \times H_2 \times D_2\\) với:
  - \\(W_2 = (W_1 - F + 2P)/S + 1\\)
  - \\(H_2 = (H_1 - F + 2P)/S + 1\\) (i.e. chiều rộng, chiều cao được coi như nhau)
  - \\(D_2 = K\\)
- Với chia sẻ tham số, nó sẽ tạo ra \\(F \cdot F \cdot D_1\\) trọng số cho mỗi filter, tổng số là \\((F \cdot F \cdot D_1) \cdot K\\) trọng số và \\(K\\) biases.
- Trong khối đầu ra, ở độ sâu thứ \\(d\\)-th  (có kích thước \\(W_2 \times H_2\\)) là kết quả thực hiện một thao tác xoắn filter thứ \\(d\\)-th qua khối đầu vào với bước kéo là \\(S\\), và có \\(d\\)-th bias là offset.

Một cấu hình phổ biến của các siêu tham số là \\(F = 3, S = 1, P = 1\\). Tuy nhiên, có những quy tắc chung và dựa trên kinh nghiệm để điều chỉnh những siêu tham số này. Xem chương [ConvNet architectures](#architectures) bên dưới.

**Convolution Demo**. Bên dưới là một demo của lớp CONV. Vì khối 3D rất khó để mô phỏng, tất cả các khối (khối đầu vào (màu xanh dương), khối trọng số (màu đỏ), khối đầu ra (màu xanh lá)) được mô phỏng theo mỗi độ sâu một dòng. Khối đầu vào có kích thước \\(W_1 = 5, H_1 = 5, D_1 = 3\\), và các tham số của lớp CONV \\(K = 2, F = 3, S = 2, P = 1\\). Đó là, chúng ta có 2 filter, với kích thước \\(3 \times 3\\), và chúng ta sử dụng bước kéo stride 2. Vì thế, kích thước không gian khối đầu ra (5 - 3 + 2)/2 + 1 = 3. Hơn nữa, chú ý là có sử dụng padding \\(P = 1\\) trên khối đầu vào, đưa vào biên của khối đầu vào một dãy 0. Mô phỏng dưới đây lần lượt thực hiện theo tức tự của các giá trị khối đầu ra (xanh lá), và chỉ ra mỗi phần tử được tính toán bằng cách nhân từng phần tử được chọn ở đầu vào (xanh da trời) và filter (đỏ), rồi cộng chúng lại, cuối cùng cọng với bias như là offset.

<div class="fig figcenter fighighlight">
  <iframe src="/assets/conv-demo/index.html" width="100%" height="700px;" style="border:none;"></iframe>
  <div class="figcaption"></div>
</div>

**Thực thi bằng phép nhân ma trận**. Chú ý rằng thao tác xoắn về cơ bản thực hiện phép nhân vô hướng giữa filter và một vùng nhỏ của đầu ra. Một cách implementation phổ biến của lớp CONV là lợi dụng tính chất này và thực hiện hàm forward pass của một lớp convolutional layer giống như một phép nhân ma trận lớn như bên dưới:

1. Các vùng cục bộ trong ảnh đầu vào được kéo giãn thành các cột thông qua một thao tác phổ biến tên là **im2col**. Ví dụ, input có kích thước [227x227x3] và được xoắn với các filte kích thước 11x11x3, theo bước kéo 4, chúng ta sẽ quan tâm đến các khối có kích thước [11x11x3] điểm trên đầu vào và kéo giãn chúng ra thành một vector cột có kích thước 11\*11\*3 = 363. Lặp lại quá trình này trên đầu vào với bước kéo là 4 sẽ cho ra (227-11)/4+1 = 55 vị trí trên cả chiều rộng và chiều cao, cuối cùng tạo được ma trận đầu ra `X_col` từ *im2col* có kích thước [363 x 3025], trong đó mõi cột là kết quả kéo dãn từ receptive field và có tổng cộng 55*55 = 3025 cột. Vì các receptive fields này bị chèn vào nhau, nên mỗi số trong khối đầu vào có thể bị lặp lại nhiều lần trong các cột khác nhau.
2. Trọng số của lớp CONV cũng tương tự nhưng được kéo giãn thành hàng. Ví dụ, có 96 filter kích thước [11x11x3] thì ta sẽ có ma trận trọng số `W_row` có kích thước [96 x 363].
3. Kết quả của phép xoắn (convolution) tương đương với việc thực hiện một phép nhân ma trận lớn `np.dot(W_row, X_col)`, sau đó có được tích vô hướng giữa filter và mỗi vị trí receptive field. Trong trường hợp của chúng ta, đầu ra của phép tính này sẽ có kích thước [96 x 3025], tức là đầu ra của phép vô hướng giữa mỗi filter trên mỗi vị trí trên ảnh đầu vào. 
4. Sau đó kết quả lại được co lại trở về kích thước mong muốn của đầu ra [55x55x96].

Hướng tiếp cận này có một điểm trừ, đó là tốn rất nhiều bộ nhớ, vì nhiều giá trị của khối đầu vào được lặp lại nhiều lần trong ma trận `X_col`. Tuy nhiên, có một lợi thế là có rất nhiều các thực thi để thực hiện phép nhân ma trận - Matrix Multiplication hiệu quả (ví dụ, một thứ được sử dụng phổ biến là [BLAS](http://www.netlib.org/blas/) API). Hơn nữa, vẫn ý tưởng sử dụng *im2col* có thể được sử dụng tiếp để thực hiện phép pooling, cái mà chúng ta sẽ thảo luận bên dưới đây.

**Phản hồi ngược.** Chiều ngược lại của thao tác xoắn (backward pass) (cho cả dữ liệu và trọng số) cũng là một convolution (nhưng filter bị đảo chiều không gian). Dễ dàng lấy đạo hàm trong trường hợp một chiều thì tương đối dễ dàng (xin không nói ở đây).

**1x1 convolution**. Một mặt khác, có vài paper sử dụng convolution kích thước 1x1, đầu tien là [Network in Network](http://arxiv.org/abs/1312.4400). Nhiều người có vẻ thấy khỏ hiểu khi thấy convolutions 1x1 đặc biệt khi họ xuất phát từ chuyên ngành xử lý tín hiệu. Thông thường các tín hiệu sẽ là 2 chiều - 2-dimensional vì thế convolutions 1x1 có vẻ không hợp lý lắm (nó chỉ là một tỉ lệ theo điểm). Tuy nhiên, trong ConvNets nó không như thế vì chúng ta thực hiện các thao tác trên khối 3-dimensional, filter sẽ luôn được mở rộng theo toàn bộ độ sâu của khối đầu vào. Ví dụ, nếu đầu vào có kích thước [32x32x3] thì khi thực hiện convolution 1x1 sẽ thực hiện phép tính vô hướng 3-dimensional (vì chiều sâu của đầu vào là 3 mà).

**Dilated convolutions.** Gần đây (ví dụ trong [paper by Fisher Yu and Vladlen Koltun](https://arxiv.org/abs/1511.07122)) đã giới thiệu thêm một tham số cho lớp CONV được gọi là sự giãn *dilation*. Cho đến giờ chúng ta chỉ thảo luận về CONV filters liền kề nhau. Tuy nhiên, có thể có filter mà có khoảng trống giữa các ô, gọi là ô nở hay dilation. Ví dụ, trong filter ở dạng một chiều `w` có kích thước 3 tính toán với đầu vào `x` bằng công thức: `w[0]*x[0] + w[1]*x[1] + w[2]*x[2]`. Đây là độ giãn nở bằng 0. Ví dụ khi độ giãn nở bằng 1, nó sẽ tính `w[0]*x[0] + w[1]*x[2] + w[2]*x[4]`; Hay nói cách khác có một gap giữa các ứng dụng. Cái này có thể rất hữu ích khi so sánh với filter không giãn - 0-dilated filters bởi vì nó cho phép bạn trộn thông tin không gian đến đầu vào mạnh hơn với số lớp ít hơn. Ví dụ, bạn đặt 2 lớp CONV kích thước 3x3 trên các lớp khác thì bạn có thể tự thuyết phụ mình rằng các neurons ở lớp thứ 2 là một hàm với đầu vào là các vùng 5x5 trên khối đầu vào (và chúng ta có thể nói rằng *effective receptive field* của những neuron này là 5x5). Nếu bạn sử dụng convolution giãn nở con số này sẽ lớn lên nhanh chóng.

<a name='pool'></a>

#### Pooling Layer

Người ta rất hay thêm lớp Pooling vào sau các lớp Conv trong cấu trúc ConvNet. Chức năng của nó là giảm số chiều không gian biểu diễn, dẫn đến giảm số lượng tham số và tính toán trong mạng, và cũng kiểm soát cả overfitting nữa. Lớp Pooling thực hiện độc lập trên mỗi độ sâu của khối đầu vào và thực hiện thay đổi kích thước hình học của nó, sử  dụng thao tác MAX. Một dạng phổ biến lớp pooling với kích thước filter bằng 2x2, bước kéo là 2 trên mỗi độ sâu ở đầu vào dọc theo chiều rộng và chiều cao, giảm đến 75% lượng kích hoạt (activations). Mỗi thao tác MAX nhận 4 số đầu vào (một khu vực nhỏ kích thước 2x2 ở mỗi độ sâu). Trường độ sâu vẫn không đổi nha. Nói chung, lớp pooling sẽ:

- Nhận một khối đầu vào với kích thước \\(W_1 \times H_1 \times D_1\\)
- Yêu cầu các siêu tham số: 
  - Phạm vi không gian \\(F\\), 
  - Bước kéo \\(S\\), 
- Tạo ra một khối với kích thước \\(W_2 \times H_2 \times D_2\\) trong đó:
  - \\(W_2 = (W_1 - F)/S + 1\\)
  - \\(H_2 = (H_1 - F)/S + 1\\)
  - \\(D_2 = D_1\\)
- Không có parameter nào hết vì nó là hàm thực hiện cố định trên đầu vào
- Chú ý là ít khi người ta zero-padding cho lớp pooling

Một điểm đáng lưu ý ở đây là, chỉ có 2 dạng phổ biến thường thấy của lớp max pooling trong thực tế: Một lớp pooling với \\(F = 3, S = 2\\) (cũng được gọi tên là pooling đè lên nhau overlapping pooling), và một dạng phổ biến hơn là \\(F = 2, S = 2\\). Pooling với kích thước lớn hơn thì gây ra mất quá nhiều thông tins.

**General pooling**. Thêm một chút về max pooling, các đơn vị pooling có thể thực hiện thêm các hàm khác, như pooling trung bình *average pooling* hoặc thậm chí *L2-norm pooling*. Pooling trung bình thường được sử dụng trước đây thôi giờ không còn được phổ biến nữa khi so sánh với max pooling, cái mà được chỉ ra là chạy tốt hơn trong thực tế.

<div class="fig figcenter fighighlight">
  <img src="/assets/cnn/pool.jpeg" width="36%">
  <img src="/assets/cnn/maxpool.jpeg" width="59%" style="border-left: 1px solid black;">
  <div class="figcaption">
    Lớp pooling giảm chiều không gian của khối, theo từng độ sâu độc lập với nhau ở đầu vào. <b>Phía trái:</b> Trong ví dụ này, kích thước khối đầu vào [224x224x64] được pooled với kích thước 2, bước kéo 2 tạo ra đầu ra có kích thước [112x112x64]. Chú ý rằng độ sâu của khối vẫn bảo toàn. <b>Phía phải:</b> Hầu hết phép giảm chiều sử dụng max, và được gọi là <b>max pooling</b>, ở đây bước kéo là 2. Hay trên đó, mỗi hàm max sẽ thực hiện trên 4 số (ô vuông nhỏ kích thước 2x2).
  </div>
</div>

**Phản hồi ngược**. Nhớ lại chương về phản hồi ngược trước đây, chiều ngược lại cho thao tác max(x, y) được diễn đạt tương đối đơn giản, chỉ cần dẫn hướng gradient đến cái đầu vào input có giá trị lớn nhất khi forward pass là xong. Cho nên, trong quá trình forward pass của pooling layer, người ta hay giữ lại index của số lớn nhất được chọn - max activation (thỉnh thoảng được gọi là *the switches*) vì thế việc dẫn hướng cho gradient khá hiệu quả khi backpropagation.

**Loại bỏ việc pooling**. Nhiều người không thích thao tác pooling và nghĩ rằng chúng ta có thể làm mà không cần nó. Ví dụ, [Striving for Simplicity: The All Convolutional Net](http://arxiv.org/abs/1412.6806) đề xuất bỏ lớp pooling để tạo ra kiến trúc chỉ chứa các lớp CONV mà thôi. Đối với việc giảm kích thước biểu diễn, họ gợi ý sử dụng một bước kéo - stride lớn hơn trong lớp CONV khi thực hiện. Việc bỏ lớp pooling cũng được phát hiện là rất quan trọng trong việc training good generative models, như variational autoencoders (VAEs) hoặc generative adversarial networks (GANs). Dường như các kiến trúc trong tương lai sẽ rất ít cho đến không có lớp pooling nữa.

<a name='norm'></a>

#### Lớp chuẩn hoat - Normalization Layer

Nhiều loại lớp chuẩn hóa được đề xuất sử dụng trong kiến trúc ConvNet, thỉnh thoảng theo hướng thực thi cái mô hình ức chế - inhibition schemes được quan sát trong bộ não sinh học. Tuy nhiên, những lớp đó vẫn chưa được để ý đến nhiều trong thực tế hiệu quả của chúng được đưa ra rất ít, nếu có. Về các loại chuẩn hóa, xem thêm thảo luận của Alex Krizhevsky's [cuda-convnet library API](http://code.google.com/p/cuda-convnet/wiki/LayerParams#Local_response_normalization_layer_(same_map)).

<a name='fc'></a>

#### Lớp kết nối đầy đủ - Fully-connected layer

Các neuron trong lớp fully connected có kết nối đến tất cả các activation ở lớp trước nó, giống như trong Neural Networks thông thường. Hàm kích hoạt của nó được tính bằng một phép nhân ma trận cộng với bias làm offset. Xem chương *Neural Network* trong cùng note để biết thêm.

<a name='convert'></a>

#### Converting FC layers to CONV layers 

Một điểm đáng lưu ý là chỉ có một điểm khác biệt duy nhất giữa lớp FC và CONV các neuron trong lớp CONV được kết nối chỉ với một vùng cục bộ ở đầu vào, và có rất nhiều neuron trong lớp CONV chia sẻ tham số với nhau. Tuy nhiên, neuron trong cả 2 lớp này đều phải tính toán tích vô hướng, vì thế về mặt chức năng thì giống nhau. Chính vì vậy, dẫn đến việc có thể chuyển đổi giữa lớp FC và CONV layers:

- Với bất cứ lớp CONV, cũng có một lớp FC thực hiện cùng một hàm forward pass. Ma trận trọng số là một ma trận lớn mà giá trị các ô hầu hết là 0 trừ một vài khối nhất định (do kết nối là cục bộ mà) rồi thì trọng số của rất nhiều khói sử dụng cùng một ma trận trọng số (do chia sẻ tham số).
- Ngược lại, mọi lớp FC có thể được chuyển đổi thành lớp CONV. Ví dụ, một lớp FC với \\(K = 4096\\), khi xem một khối đầu vào có kích thước \\(7 \times 7 \times 512\\) có thể tương đương với một lớp CONV với \\(F = 7, P = 0, S = 1, K = 4096\\). Nói cách khác, chúng ta thiết lập kích thước filter đúng bằng kích thước của đầu vào luôn, đầu ra sẽ đơn giản có kích thước \\(1 \times 1 \times 4096\\) vì mỗi độ sâu chỉ chứa kế quả xoắn với đầu vào đúng 1 lần, tức là kết quả tương lớp FC luôn.

**FC->CONV conversion**. Trong 2 loại chuyển đổi, việc chuyển từ lớp FC sang lớp CONV đặc biệt hữu ích trong thực tế. Xét một kiến trúc ConvNet lấy đầu vào là ảnh 224x224x3, và sử dụng một loạt lớp CONV và POOL về thành một khối kích hoạt có kích thước 7x7x512 (trong kiến trúc *AlexNet* chúng ta sẽ thấy sau đây, sử dụng 5 lớp pooling giảm không gian đầu vào xuống theo hệ số 2 mỗi lần, để cuối cùng không gian kết quả có số chiều 224/2/2/2/2/2 = 7). Từ đây, AlexNet sử dụng 2 lớp FC kích thước 4096 sau đó là lớp FC cuối cùng có 1000 neurons để tính toán core cho từng lớp. Chúng ta có thể chuyển mỗi lớp trong 3 lớp FC cuối sang lớp CONV tương ứng theo mô tả dưới đây:

- Thay thế lớp FC đầu tiên cái lấy khối [7x7x512] làm đầu vào bằng một lớp CONV có kích thước filter \\(F = 7\\), sẽ tạo ra khối đầu ra có kích thước [1x1x4096].
- Thay thế lớp FC thứ 2 bằng một lớp CONV với kích thước filter \\(F = 1\\), sẽ tạo ra khối đầu ra kích thước [1x1x4096]
- Thay thế lớp FC cuối cùng cũng tương tự, với \\(F=1\\), và tạo ra khối đầu ra cuối cùng [1x1x1000]

Mỗi phép chuyển đổi ở trên trong thực tế sẽ thực hiện thao tác (ví dụ: reshaping) với ma trận trọng số \\(W\\) trong mỗi lớp FC thành filter trong lớp CONV tương ứng. Điều này chỉ ra rằng các phép chuyển đổi này cho phép chúng ta "kéo" kiến trúc ConvNet một cách hiệu quả qua nhiều vị trí không gian khác nhau trong trong một ảnh lớp hơn, chỉ trong một phép forward pass thôi. 

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
