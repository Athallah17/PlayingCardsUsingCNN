import ModulKlasifikasiCitraCNN2 as mCNN

# Nama folder datasetnya
DirektoriDataSet = "CardDataSet"

JumlahEpoh = 3

# Label kelas disini harus sesuai dengan nama folder didalam folder CardDataSet

cardName = (
  "Closed Card","Two Club","Three Club","Four Club","Five Club","Six Club","Seven Club","Eight Club","Nine Club","Ten Club","Jack Club","Queen Club","King Club","Ace Club",
  "Two Heart","Three Heart","Four Heart","Five Heart","Six Heart","Seven Heart","Eight Heart","Nine Heart","Ten Heart","Jack Heart","Queen Heart","King Heart","Ace Heart",
  "Two Spade","Three Spade","Four Spade","Five Spade","Six Spade","Seven Spade","Eight Spade","Nine Spade","Ten Spade","Jack Spade","Queen Spade","King Spade","Ace Spade",
  "Two Diamonds","Three Diamonds","Four Diamonds","Five Diamonds","Six Diamonds","Seven Diamonds","Eight Diamonds","Nine Diamonds","Ten Diamonds","Jack Diamonds","Queen Diamonds","King Diamonds","Ace Diamonds",
  )
# Mulai training
mCNN.TrainingCNN(JumlahEpoh, DirektoriDataSet, cardName,"CardModelWeight.h5")