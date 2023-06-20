# car and lincense plate detection project

## description(in persian)
<div dir="rtl">
<h4>مقدمه</h4>
در این پروژه از دو مدل YOLOv5 و YOLO NAS برای تشخیص خودرو و پلاک خودرو در ویدیو است.
از YOLO NAS، که بر پایه ی ساختار YOLO است برای تشخیص خودرو استفاده شده است. به لطف کلاس و وزن های از پیش train شده بر روی دیتاست COCO، تشخیص خودرو با YOLO NAS کار آسانی خواهد بود. از طرفی، مدل YOLOv5، با استفاده از یک custom dataset پلاک های خودرو از صفر train شده است.

 <h4>بررسی مدل</h4>
ساختار (YOLO(you only look once یک سیستم تشخیص اشیا است که یک عکس را به عنوان ورودی می گیرد و آن را به یک grid تقسیم بندی می کند. هر گرید سل، bounding box ها و درصد احتمال کلاس ها را پیش بینی می کند.
YOLO از یک شبکه ی عصبی برای پیش بینی bounding box ها و درصد احتمال کلاس ها به صورت همزمان استفاده می کند.
 
![0_WUpMWzNu_ymDyHPp](https://github.com/dev-parsa/object-detection/assets/105069707/32dcd8b7-5627-45b4-b9e3-b9dd4a7a971c)
 
 <h4>چه چیزی YOLO را برای تشخیص شی جذابتر از دیگر مدل ها می کند؟</h4>
YOLO بسیار سریع است. شاید سریع بودن این مدل یکی از بزرگ ترین مزیت های آن باشد. در YOLO تنها یک شبکه عصبی وجود دارد که خیلی ساده به آن ورودی تصویر داده می‌شود تا شبکه پیش‌بینی‌های تشخیص اشیا را به ما نشان دهد. سرعت YOLO باعث شده که در اپلیکیشن های real-time هم از آن بهره گرفته شود
YOLO برای تشخیص، به صورت کلی (Global) به تصویر نگاه می‌کند. برخلاف تکنیک‌های پنجره‌های لغزان (sliding window) و پروپوزال، YOLO به کل تصویر نگاه می‌کند.

<h4 dir="rtl">YOLO NAS در YOLO بهبود مدل</h4>
استفاده از بلوک های QSP و QCI مزایای پارامترسازی مجدد و کوانتیزاسیون 8 بیتی را ترکیب می کند. بلوک ها از دست دادن دقت را در طول کوانتیزاسیون پس از آموزش حداقل  می کنند.<br/>
فناوری NAS اختصاصی Deci، AutoNAC، برای تعیین اندازه و ساختار بهینه مراحل، از جمله نوع بلوک، تعداد بلوک‌ها و تعداد کانال‌ها در هر مرحله استفاده شد.<br/>
<br/>
یک روش کوانتیزاسیون ترکیبی که به طور انتخابی بخش‌های خاصی از یک مدل را کوانتیزه می‌کند، از دست دادن اطلاعات را کاهش می‌دهد و تأخیر و دقت را متعادل می‌کند. کوانتیزاسیون همه لایه‌های مدل را تحت تأثیر قرار می‌دهد و اغلب منجر به کاهش دقت قابل توجهی می‌شود. روش ترکیبی، کوانتیزاسیون را برای حفظ دقت تنها با کم کردن لایه‌های خاصی بهینه می‌کند و در عین حال بقیه را دست نخورده می‌گذارد.<br/>
<br/>
یافتن معماری "درست" با آزمون و خطا بسیار خسته کننده و ناکارآمد است. بنابراین،  از AutoNAC برای کشف مدل‌های جدید تشخیص اشیاء بهینه‌سازی شده برای به حداقل رساندن تأخیر محاسبه‌شده روی T4 NVIDIA - یک پردازنده گرافیکی ابری پرکاربرد استفاده شده است.<br/>
<br/>
الگوریتم‌های NAS می‌توانند به طور سیستماتیک فضای جستجوی وسیع معماری‌های ممکن را سرچ کنند، و به طور موثر پیکربندی‌های جدید و بهینه‌شده‌ای را که ممکن است توسط شهود انسان نادیده گرفته شوند، شناسایی کنند. با خودکار کردن فرآیند، این الگوریتم‌ها می‌توانند به طور موثر تعداد زیادی از معماری‌های نامزد را ارزیابی و مقایسه کنند و در نهایت روی راه‌حلی همگرا شوند که دقت، سرعت و پیچیدگی را به طور بهینه متعادل کند.<br/>
<br/>
در چارت زیر یک مقایسه از مدل YOLO NAS با مدل های پیشین YOLO آمده است.

![yolo_nas_rf100](https://github.com/dev-parsa/object-detection/assets/105069707/a9e4085f-2985-4d0c-a87e-d0f4cbecc805)

![yolo_nas_frontier](https://github.com/dev-parsa/object-detection/assets/105069707/54568a90-ad0c-4942-a899-17e0dd2c490f)




  <h4>تشخیص پلاک خودرو</h4>
برای تشخیص پلاک خودرو در ویدیو، برای train مدل و fine tune کردن مدل yolov5 از یک دیتاست car license plate detection از سایت kaggle استفاده شده است. این دیتاست تقریبا دارای 400 عکس به همراه annotation آنها است. Yolov5 از فرمت txt خاصی استفاده می کند به همین دلیل باید فایل های xml را به فرمت txt  مورد قبول yolo تبدیل کنیم.
برای آموزش به مدل، batch به اندازه 32، yolov5 large، image size 320 و 100 تا epoch در نظر گرفنه شده است.
همچنین برای سرعت دادن به مرحله ی آموزش، از GPU بهره گرفته شده است.
سایر توضیحات در مورد fine tuning مدل داخل فایل notebook وجود دارد.
برخی از Metric های بدست آمده بصورت زیر است.

  ![results](https://github.com/dev-parsa/object-detection/assets/105069707/b0ca7aa9-db12-4629-adce-4aaee33b345b)

  ![labels](https://github.com/dev-parsa/object-detection/assets/105069707/14b6e7b9-0a6d-45b4-8261-a718b20e3ad4)

  <h4>تشخیص خودرو</h4>
 برای تشخیص خودرو از مدل YOLO NAS استفاده شده است. از آنجایی یکی از 80 کلاس از پیش تعریف شده ی coco ماشین است و YOLO NAS قابلیت استفاده از pretrained weights را در اختیار ما قرار داده است، نیازی به fine tune کردن مدل وجود ندارد. 
برای detect خودرو در ویدیو از مدل YOLO NAS large استفاده شده است.

  

</div>

## pretrained weights

If you'd like to experiment or if you prefer not to train the model from scratch for license plate detection, feel free to utilize the pretrained weights provided in "my_weights/last.pt". These weights are obtained from the model training process, and can be used as a starting point for your own license plate detection tasks.

## installation

Installing dependencies for YOLO NAS is a relatively straightforward task. You just need to run the code cells to install the required packages. However, it's a bit different when it comes to YOLOv5. Here's what you need to do to install the dependencies for YOLOv5 to work:

1. Clone the YOLOv5 repository from GitHub:
```
!git clone https://github.com/ultralytics/yolov5.git
```
2.Navigate to the yolov5 directory:
```
%cd yolov5
```
3.Install the required dependencies using pip:
```
!pip install -r requirements.txt
```
This will install all the necessary packages specified in the requirements.txt file.
Now you should be good to go :)




