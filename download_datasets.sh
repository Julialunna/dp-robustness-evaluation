datasets=(
  pathmnist_224
  octmnist_224
  bloodmnist_224
  tissuemnist_224
  organamnist_224
  organcmnist_224
  organsmnist_224
)

for dataset in "${datasets[@]}"; do
    hf download "danjacobellis/$dataset" \
        --repo-type dataset \
        --local-dir "$dataset"

    zip -r "${dataset}.zip" "$dataset"
done

oi, pessoal!!
passando aqui para convidar vocês para comemorar meu aniversário na sexta com um churrasquinho
horário: 20h
espero todos vocês!!