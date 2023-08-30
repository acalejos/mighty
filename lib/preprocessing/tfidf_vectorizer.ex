defmodule Mighty.Preprocessing.TfidfVectorizer do
  @moduledoc ~S"""
  Term Frequency Inverse Document Frequency Vectorizer for `Mighty`.

  ## Examples
  ```elixir
  iex> corpus = [
  ...>   "This is the first document",
  ...>   "This document is the second document",
  ...>   "And this is the third one",
  ...>   "Is this the first document"
  ...> ]
  iex> vocabulary = ["this", "document", "first", "is", "second", "the",
  ...>               "and", "one"]
  iex> vectorizer = Mighty.Preprocessing.TfidfVectorizer.new(corpus, vocabulary: vocabulary, norm: 2)
  iex> vectorizer.count_vectorizer.vocabulary |> Map.keys() |> Enum.sort()
  ["and", "document", "first", "is", "one", "second", "the", "this"]
  iex> {tf, _df} = vectorizer.count_vectorizer |> Mighty.Preprocessing.CountVectorizer.transform(corpus)
  iex> tf |> Nx.to_list()
  [
    [0, 1, 1, 1, 0, 0, 1, 1],
    [0, 2, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0, 1, 1]
  ]
  iex> Mighty.Preprocessing.TfidfVectorizer.transform(vectorizer, corpus) |> Nx.to_list()
  [
    [0.0, 0.3364722366212129, 0.3364722366212129, 0.3364722366212129, 0.0, 0.0, 0.3364722366212129, 0.3364722366212129],
    [0.0, 0.6731340946124258, 0.0, 0.22437803153747527, 0.0, 0.22437803153747527, 0.22437803153747527, 0.22437803153747527],
    [0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.4082482904638631, 0.0, 0.4082482904638631, 0.4082482904638631],
    [0.0, 0.3364722366212129, 0.3364722366212129, 0.3364722366212129, 0.0, 0.0, 0.3364722366212129, 0.3364722366212129]
  ]
  ```
  """
  alias Mighty.Preprocessing.CountVectorizer
  alias Mighty.Preprocessing.Shared

  defstruct [
    :count_vectorizer,
    use_idf: true,
    smooth_idf: true,
    sublinear_tf: false,
    norm: nil
  ]

  def new(corpus, opts \\ []) do
    {general_opts, tfidf_opts} =
      Keyword.split(opts, Shared.get_vectorizer_schema() |> Keyword.keys())

    count_vectorizer = CountVectorizer.new(corpus, general_opts)
    tfidf_opts = Shared.validate_tfidf!(tfidf_opts)

    %__MODULE__{count_vectorizer: count_vectorizer}
    |> struct(tfidf_opts)
  end

  # def transform(%Nx.Tensor{} = tf, %Nx.Tensor{} = df, opts \\ []) do
  #   tfidf_opts = Shared.validate_tfidf!(tfidf_opts)
  # end
  # TODO: Allow passing TF+DF matrix or just Counts matrix -- add ability to directly use transform
  def transform(%__MODULE__{count_vectorizer: count_vectorizer} = vectorizer, corpus) do
    {tf, df} = CountVectorizer.transform(count_vectorizer, corpus)

    idf =
      if vectorizer.use_idf do
        {n_samples, _n_features} = Nx.shape(tf)
        df = Nx.add(df, if(vectorizer.smooth_idf, do: 1, else: 0))
        n_samples = if vectorizer.smooth_idf, do: n_samples + 1, else: n_samples
        Nx.divide(n_samples, df) |> Nx.log() |> Nx.add(1) |> Nx.make_diagonal() |> dbg
      end

    tf =
      if vectorizer.sublinear_tf do
        Nx.log(tf) |> Nx.add(1)
      else
        tf
      end

    tf =
      if vectorizer.use_idf do
        Nx.dot(tf, idf)
      else
        tf
      end

    # TODO: Handle Normalization
    # (https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-normalization)
    # (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html#sklearn.preprocessing.normalize)

    tf
  end
end
