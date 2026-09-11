# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):

class_id: 468
class_name: cab
rank: 1
score: `0.510915`
taxonomy_name: `ImageNet-1K`

- Record này mô tả toàn ảnh như thế nào?

Mô hình dự đoán toàn ảnh traffic có khả năng cao nhất thuộc class cab, với class_id: 468, hạng 1 và điểm 0.510915. Đây là dự đoán ở cấp độ toàn ảnh không cho biết vị trí hay số lượng từng vật thể.

- Ai định nghĩa class list mà checkpoint có thể dự đoán?

Class list do **taxonomy và bộ dữ liệu dùng để huấn luyện checkpoint** định nghĩa. Trong trường hợp này là taxonomy  **ImageNet-1K** , được checkpoint `yolo11n-cls.pt` sử dụng.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?

`class_id` là mã ổn định để máy xử lý và đối chiếu.
`class_name` giúp con người đọc và kiểm tra kết quả.
`taxonomy_name` cho biết lớp đó thuộc hệ phân loại nào, tránh nhầm khi các taxonomy khác nhau có thể dùng cùng tên hoặc ID khác nhau.
Ba trường này giúp kết quả có thể truy vết, tái lập và tích hợp chính xác.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?

Guideline cần quy định rõ mục tiêu của nhãn: chọn  **chủ thể chính** , chọn  **một lớp đại diện cho toàn cảnh** , hay cho phép  **nhiều nhãn** . Cũng cần nêu tiêu chí xử lý khi các chủ thể có tầm quan trọng tương đương, khi chủ thể bị che khuất hoặc khi ảnh không đủ rõ; các trường hợp mơ hồ nên được đánh dấu để reviewer/escalation quyết định.

- Vì sao model score không phải ground truth?

`score` chỉ thể hiện mức độ mô hình tin vào dự đoán dựa trên dữ liệu và cách mô hình được huấn luyện. Mô hình có thể dự đoán sai nhưng vẫn có score cao, hoặc dự đoán đúng với score thấp. Ground truth phải được xác định từ quy tắc gán nhãn và quá trình annotation/QC của con người, không thể suy ra chỉ từ score.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):

`class_name`: `person`
`score`:`0.912625`
`bbox_xyxy`: 385.33, 69.24, 498.92, 348.92
`bbox_width`: `113.58`
`bbox_height`: `279.68`

- Diễn giải vị trí box bằng lời:

Với prediction person trong ảnh kitchen, box có tọa độ [385.33, 69.24, 498.92, 348.92] theo định dạng xyxy và đơn vị pixel. Box nằm ở khu vực bên phải, bên cạnh box oven, kéo dài từ gần phía trên đến khoảng giữa thấp của ảnh, có chiều rộng 113.58 px và chiều cao 279.68 px. Box không chạm mép ảnh.

- So sánh số prediction ở hai threshold:

Threshold 0.2 có 17 vật thể
Threshold 0.6 có 6 vật thể

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?

Threshold thấp giữ lại nhiều prediction hơn, giúp tăng khả năng bao phủ các object nhỏ, xa, bị che khuất hoặc khó nhận diện, nhưng reviewer phải kiểm tra nhiều hơn và có thể gặp thêm false positive. Threshold cao giảm số prediction và khối lượng review, nhưng dễ bỏ sót object thật có score thấp.

- Đề xuất một quy tắc box chặt:

Box phải bao quanh sát toàn bộ phần object nhìn thấy, không lấy thêm nền không cần thiết và không cắt vào phần object đang quan sát được. Mỗi object riêng biệt cần một box riêng; tọa độ phải hợp lệ trong phạm vi ảnh.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?

Guideline cần quy định có gán nhãn khi object chỉ nhìn thấy một phần hay không, mức độ tối thiểu được phép gán, và box bao quanh phần nhìn thấy hay ước đoán toàn bộ object. Object bị cắt ở mép ảnh nên được ghi nhận rõ là truncated; object bị che khuất nên có quy tắc về mức độ visibility. Những trường hợp không xác định được lớp hoặc phạm vi box nên được đánh dấu để reviewer/escalation quyết định.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`)

`instance_id`: `kitchen-001`
`class_name`: `person`
`score`: `0.899318`
`polygon_point_count`: `348`
`polygon_xy`: danh sách 348 điểm `[x, y]` tạo thành đường biên của người trong ảnh.

```json
[
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
```

- Polygon bổ sung chi tiết gì so với box?

Box chỉ là hình chữ nhật bao quanh object, nên có thể chứa cả nền. Polygon mô tả sát đường viền thực tế của object, thể hiện hình dạng không vuông vức, phần lõm, tay/chân hoặc các vùng biên phức tạp. Vì vậy polygon phù hợp hơn khi cần biết chính xác pixel nào thuộc object.

- `instance_id` dùng để làm gì và không phải loại ID nào?

`instance_id` dùng để nhận diện duy nhất từng object cụ thể trong output, liên kết object đó với class, score, box và polygon. Hai object cùng lớp vẫn có ID khác nhau, như `kitchen-002` và `kitchen-003`. Nó **không phải** `class_id`, không phải mã lớp COCO, không phải `coco_image_id` và cũng không phải ground-truth ID do người gán nhãn xác nhận.

- Đề xuất một quy tắc biên mask:

Mask phải bao phủ phần pixel thuộc về object nhìn thấy, bám sát biên ngoài thực tế, không lấy nền và không để hở phần object rõ ràng. Với các chi tiết rất nhỏ hoặc biên mờ, annotator cần tuân theo ví dụ chuẩn trong guideline thay vì tự thay đổi mức độ chi tiết giữa các ảnh.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?

Guideline cần quyết định:

1. Có gán mask cho object chỉ nhìn thấy một phần hay không.
2. Mask chỉ bao quanh phần nhìn thấy hay được phép suy đoán phần bị che.
3. Cách xử lý bóng, phản chiếu, nền có màu tương tự và vùng rìa mờ.
4. Khi hai object tiếp xúc hoặc chồng lấn, mỗi instance có mask riêng và ranh giới được đặt ở đâu.

Nếu không xác định được biên thật, mức che khuất quá lớn hoặc không phân biệt được object với nền, annotator nên đánh dấu để reviewer/escalation quyết định.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ              | Đơn vị/định dạng ground truth                                                                                 | Lỗi hoặc điểm mơ hồ quan sát được                                                                                                                                                                                                                                                                                                    | Annotator làm gì?                                                                                                        | Reviewer xem gì?                                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh      | Một nhãn lớp cho toàn ảnh, gồm`class_id` và `class_name` theo taxonomy đã quy định                   | Ảnh`traffic` có nhiều phương tiện nhưng mô hình chọn lớp `cab` với score chỉ `0.510915`; dự đoán cấp ảnh không cho biết vật thể nào là chủ thể chính và score không đảm bảo đúng ground truth.                                                                                                         | Chọn lớp theo chủ thể hoặc mục tiêu chính được guideline quy định; đánh dấu ảnh mơ hồ để xử lý      | Kiểm tra lớp có phù hợp toàn ảnh, đúng taxonomy và nhất quán với guideline không                      |
| Phát hiện vật thể | Một record cho mỗi object bao gồm nhưng không giới hạn`class_id`, `class_name`, `bbox_xyxy` theo pixel | Có nhiều prediction score thấp, chẳng hạn một số object trong ảnh`kitchen` gần ngưỡng `0.35`; có nguy cơ false positive hoặc bỏ sót object nhỏ, bị che khuất. Một số box chạm mép ảnh nên cần quy tắc xử lý object bị cắt.                                                                                  | Gán đủ từng object nhìn thấy, vẽ box sát phần nhìn thấy, ghi nhận trường hợp bị che khuất hoặc cắt mép | Kiểm tra đủ object, đúng lớp, box không chứa quá nhiều nền, không bị trùng hoặc bỏ sót             |
| Instance segmentation | Một mask/polygon riêng cho mỗi instance, kèm`instance_id`, `class_id`, `polygon_xy`,...                   | Một số polygon có ít điểm và score thấp, ví dụ`kitchen-011` có `37` điểm với score `0.359146`; biên mask có thể chưa chính xác ở object nhỏ. Các object chạm hoặc vượt sát mép ảnh, như `potted plant`, `oven` hoặc `dining table`, cần quy định rõ mask phần nhìn thấy và phần bị cắt. | Vẽ mask theo phần object nhìn thấy, giữ instance riêng và đánh dấu trường hợp không rõ biên                | Kiểm tra mask bám sát biên object, không lẫn nền, các instance không bị gộp và`instance_id` duy nhất |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Reviewer

## 6. Danh sách bằng chứng

- [X] `classification_predictions.json`
- [X] `detection_predictions.json`
- [X] `segmentation_predictions.json`
- [X] `IMAGE_ATTRIBUTION.md`
- [X] `visuals/classification_top5.png`
- [X] `visuals/detection_predictions.png`
- [X] `visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`.
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
