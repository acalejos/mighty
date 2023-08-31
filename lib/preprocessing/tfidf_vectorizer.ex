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
  iex> vectorizer = Mighty.Preprocessing.TfidfVectorizer.new(corpus, vocabulary: vocabulary, norm: nil)
  iex> vectorizer.count_vectorizer.vocabulary |> Map.keys() |> Enum.sort()
  ["and", "document", "first", "is", "one", "second", "the", "this"]
  iex> tf = vectorizer.count_vectorizer |> Mighty.Preprocessing.CountVectorizer.transform(corpus)
  iex> tf |> Nx.to_list()
  [
    [0, 1, 1, 1, 0, 0, 1, 1],
    [0, 2, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0, 1, 1]
  ]
  iex> Mighty.Preprocessing.TfidfVectorizer.transform(vectorizer, corpus) |> Nx.to_list()
  [
    [0.0, 1.2231435775756836, 1.5108256340026855, 1.0, 0.0, 0.0, 1.0, 1.0],
    [0.0, 2.446287155151367, 0.0, 1.0, 0.0, 1.9162907600402832, 1.0, 1.0],
    [1.9162907600402832, 0.0, 0.0, 1.0, 1.9162907600402832, 0.0, 1.0, 1.0],
    [0.0, 1.2231435775756836, 1.5108256340026855, 1.0, 0.0, 0.0, 1.0, 1.0]
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

  def transform(source, opts_or_corpus \\ [])

  def transform(%Nx.Tensor{} = tf, opts) do
    opts = Shared.validate_tfidf!(opts)
    df = Scholar.Preprocessing.binarize(tf) |> Nx.sum(axes: [0])

    idf =
      if opts[:use_idf] do
        {n_samples, _n_features} = Nx.shape(tf)
        df = Nx.add(df, if(opts[:smooth_idf], do: 1, else: 0))
        n_samples = if opts[:smooth_idf], do: n_samples + 1, else: n_samples
        Nx.divide(n_samples, df) |> Nx.log() |> Nx.add(1)
      end

    tf =
      if opts[:sublinear_tf] do
        Nx.select(Nx.equal(tf, 0), 0, Nx.log(tf) |> Nx.add(1))
      else
        tf
      end

    tf =
      if opts[:use_idf] do
        Nx.multiply(tf, idf)
      else
        tf
      end

    tf =
      case opts[:norm] do
        nil -> tf
        norm -> Scholar.Preprocessing.normalize(tf, norm: norm)
      end

    tf
  end

  def transform(%__MODULE__{count_vectorizer: count_vectorizer} = vectorizer, corpus) do
    tf = CountVectorizer.transform(count_vectorizer, corpus)

    opts = Map.from_struct(vectorizer) |> Enum.into([])

    {_general_opts, tfidf_opts} =
      Keyword.split(
        opts,
        Shared.get_vectorizer_schema() |> Keyword.keys() |> Enum.concat([:count_vectorizer])
      )

    transform(tf, tfidf_opts)
  end
end
