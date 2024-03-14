import cv2
import os
import datetime
import time
# import ModulKlasifikasiCitraCNN as mCNN
import numpy as np

# Untuk penamaan semua class di Model ML
cardName =[
  "Closed Card","Two Club","Three Club","Four Club","Five Club","Six Club","Seven Club","Eight Club","Nine Club","Ten Club","Jack Club","Queen Club","King Club","Ace Club",
  "Two Heart","Three Heart","Four Heart","Five Heart","Six Heart","Seven Heart","Eight Heart","Nine Heart","Ten Heart","Jack Heart","Queen Heart","King Heart","Ace Heart",
  "Two Spade","Three Spade","Four Spade","Five Spade","Six Spade","Seven Spade","Eight Spade","Nine Spade","Ten Spade","Jack Spade","Queen Spade","King Spade","Ace Spade",
  "Two Diamonds","Three Diamonds","Four Diamonds","Five Diamonds","Six Diamonds","Seven Diamonds","Eight Diamonds","Nine Diamonds","Ten Diamonds","Jack Diamonds","Queen Diamonds","King Diamonds","Ace Diamonds",
  ]

# Kita mulai dari index 0
cardNameIndex = 0

# Fungsi
def GetFileName():
    x = datetime.datetime.now()
    s = x.strftime('%Y-%m-%d-%H%M%S%f')
    return s

# Fungsi 
def polygon_area(points):
    # 'points' adlh input berupa array yg berisikan 4 titik koordinat kartu
    points = np.vstack((points, points[0]))
    area = 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))
    return area

# Fungsi
def CreateDir(path):
    ls = []
    head_tail = os.path.split(path)
    ls.append(path)
    while len(head_tail[1])>0:
        head_tail = os.path.split(path)
        path = head_tail[0]
        ls.append(path)
        head_tail = os.path.split(path)
    for i in range(len(ls)-2,-1,-1):
        sf = ls[i]
        isExist = os.path.exists(sf)
        if not isExist:
            os.makedirs(sf)

# Fungsi 
def CreateDataSet(sDirektoriData,sKelas,NoKamera,FrameRate):
    global cardName, cardNameIndex

    # For webcam input:
    cap = cv2.VideoCapture(2)
    TimeStart = time.time()

    # Ini buat ngelimit 1 data kartu ambil berapa detik
    saveTimeLimit = time.time()

    # For start taking pics
    isSaving = False
        
    while cap.isOpened():
        success, frame = cap.read()
        
        # Buat dulu folder sesuai dengan datasetnya
        sDirektoriKelas = sDirektoriData+"/"+cardName[cardNameIndex]
        CreateDir(sDirektoriKelas)

        if not success:
            print("Ignoring empty camera frame.")
            continue

        isDetected = False

        # Image processing
        # ======= 1. ======= Pertama, gambar kita buat grayscale dulu
        # Gray
        imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("1. Gray scale dulu", imGray)

        # ======= 2. ======= Kedua, kita threshold buat ambil
        # Threshold
        #                                                    Nilai 71 dan 10 bisa diatur sesuai kebutuhan masing2. Caranya? Cari aja di google 71 itu apa 10 itu apa. Kalo udah coba2 nilai yg pas buat kamera dan kartu kalian
        # NOTES:                                                                                        v   v    
        imThres = cv2.adaptiveThreshold(imGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,65,10)
        cv2.imshow("2. Adaptive thres", imThres)

        # ======= 3. ======= Next, kita ambil component yg connected.
        # Ini diambil dari contoh nya pak Akok di catatan buat HSV, cuma diedit dikit2
        # https://drive.google.com/file/d/1nuPMCajNSBYXBYO4t5nESIi74eNIzb1b/view
        # Connecting
        totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(imThres, 4, cv2.CV_32S)
        # Big index adalah array untuk menampung index mana yg luas area connected componentnya sesuai keinginan kita
        bigIndex = []
        for i in range(totalLabels):
            hw = values[i,2:4]
            # 100 dan 300 itu untuk cari widht diantara 100-300, 
            # 300 dan 500 itu cari height antara 300-500
            # Harus dicoba2 biar hasilnya sesuai dengan kartu kalian
            # Cara nyoba gimana? Kalo gambar kartunya belom di kotakin, berarti nilainya masih salah. Tweeking aja coba2
            if (100<hw[0]<300 and 100<hw[1]<300):
                bigIndex.append(i)

        # ======= 4. ======= Check, apakah ada connected component yg sesuai dengan luas yg kita define
        # Kalo ada kita kotakin trus kita kotakin
        for i in bigIndex:
            topLeft = values[i,0:2]
            bottomRight = values[i,0:2]+values[i,2:4]
            # v                     v       Disini aku ngotakin di gambar asli 'frame'
            frame = cv2.rectangle(frame, topLeft, bottomRight, color=(0,0,255), thickness=3)

            # Disini ada break, yg berarti kita cuma ngambil 1 item doang
            break
        # Trus tampilin
        cv2.imshow("4. Hasil habis dikotakin", frame)
        
        # ======= 5. ======= Kita crop yg dikotakin tadi
        for i in bigIndex:
            topLeft = values[i,0:2]
            bottomRight = values[i,0:2]+values[i,2:4]
            
            # Ini buat ngecrop gambarnya
            cardImage = imThres[topLeft[1]:bottomRight[1],topLeft[0]:bottomRight[0]]
            
            # Lanjut ditampilin
            cv2.imshow('5. Hasil dari cardImage', cardImage)
            # Gambar ini yg bakal disimpen dan masuk jadi dataset model
            
            # Lagi lagi cuma ngambil 1 item doang
            break
        
        # ======= 6. ======= Ini buat ngerecord datasetnya.
        # Diambil juga dari modul bapaknya, yg dimodif dikit
        TimeNow = time.time()
        if TimeNow-TimeStart>1/FrameRate:
            sfFile = sDirektoriKelas+"/"+GetFileName()
            # Kita bakal nyimpen kalo sudah teken spasi, dan kal0 ada kartu yg terdeteksi
            if isSaving and len(bigIndex) > 0:
                cv2.imwrite(sfFile+'.jpg', cardImage)
            TimeStart = TimeNow

        # Buat Kata2 doang
        cv2.putText(frame, "Nama Kartu yg direkam:", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, f"{cardNameIndex+1}. " + cardName[cardNameIndex], (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Tekan spasi untuk mulai record", (0, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        # Buat visualisasi record
        if isSaving:
            cv2.putText(frame, "Record", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            saveTimeLimit = time.time()

        cv2.imshow("Tampilan akhir", frame)
        
        key = cv2.waitKey(5)

        # Trigger tekan spasi untuk mulai menyimpan gambar
        if key == 32:
            isSaving = not isSaving

        # Kalo udah lebih dari 5 detik, penyimpanan foto selesai
        # Bisa edit sesuai kebutuhan brp detiknya
        if time.time() - saveTimeLimit >= 5:
            cardNameIndex += 1
            isSaving = False

        if key & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# Nama Parent Folder dimana dataset akan ditaruh
DirektoriDataSet = "CardDataSet"

# Kita panggil fungsinya sekali aja, nanti akan looping sendiri
CreateDataSet(DirektoriDataSet, "Kosong Aja udah ini, nanti kan di replace", NoKamera=0, FrameRate=20) 