defmodule Mighty.Preprocessing.TfidfVectorizer do
  @moduledoc ~S"""
  Term Frequency Inverse Document Frequency Vectorizer for `Mighty`.

  ## Examples
  ```elixir
  iex> alias Mighty.Preprocessing.TfidfVectorizer, as: TV
  iex> corpus = [
  ...>   "This is the first document",
  ...>   "This document is the second document",
  ...>   "And this is the third one",
  ...>   "Is this the first document"
  ...> ]
  iex> vocabulary = ["this", "document", "first", "is", "second", "the",
  ...>               "and", "one"]
  iex> vectorizer = TV.new(vocabulary: vocabulary, norm: nil) |> TV.fit(corpus)
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
  iex> TV.transform(vectorizer, corpus) |> Nx.to_list()
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
    :norm,
    :idf,
    use_idf: true,
    smooth_idf: true,
    sublinear_tf: false
  ]

  @doc """
  Creates a new `TfidfVectorizer` struct with the given options.

  Returns the new vectorizer.
  """
  def new(opts \\ []) do
    {general_opts, tfidf_opts} =
      Keyword.split(opts, Shared.get_vectorizer_schema() |> Keyword.keys())

    count_vectorizer = CountVectorizer.new(general_opts)
    tfidf_opts = Shared.validate_tfidf!(tfidf_opts)

    %__MODULE__{count_vectorizer: count_vectorizer}
    |> struct(tfidf_opts)
  end

  @doc """
  Fits the vectorizer to the given corpus, computing its Term-Frequency matrix and its Inverse Document Frequency.

  Returns the fitted vectorizer.
  """
  def fit(%__MODULE__{count_vectorizer: count_vectorizer} = vectorizer, corpus) do
    {cv, tf} = CountVectorizer.fit_transform(count_vectorizer, corpus)
    df = Scholar.Preprocessing.binarize(tf) |> Nx.sum(axes: [0])

    idf =
      if vectorizer.use_idf do
        {n_samples, _n_features} = Nx.shape(tf)
        df = Nx.add(df, if(vectorizer.smooth_idf, do: 1, else: 0))
        n_samples = if vectorizer.smooth_idf, do: n_samples + 1, else: n_samples
        Nx.divide(n_samples, df) |> Nx.log() |> Nx.add(1)
      end

    struct(vectorizer, count_vectorizer: cv, idf: idf)
  end

  @doc """
  Transforms the given corpus into a Term-Frequency Inverse Document Frequency matrix.

  If the vectorizer has not been fitted yet, it will raise an error.

  Returns the TF-IDF matrix of the corpus given a prior fitting.
  """
  def transform(%__MODULE__{count_vectorizer: count_vectorizer} = vectorizer, corpus) do
    tf = CountVectorizer.transform(count_vectorizer, corpus)

    tf =
      if vectorizer.sublinear_tf do
        Nx.select(Nx.equal(tf, 0), 0, Nx.log(tf) |> Nx.add(1))
      else
        tf
      end

    tf =
      if vectorizer.use_idf do
        unless vectorizer.idf do
          raise "Vectorizer has not been fitted yet. Please call `fit_transform` or `fit` first."
        end

        Nx.multiply(tf, vectorizer.idf)
      else
        tf
      end

    tf =
      case vectorizer.norm do
        nil -> tf
        norm -> Scholar.Preprocessing.normalize(tf, norm: norm)
      end

    tf
  end

  @doc """
  Fits the vectorizer to the given corpus and transforms it into a Term-Frequency Inverse Document Frequency matrix.

  Returns a tuple with the fitted vectorizer and the transformed corpus.
  """
  def fit_transform(%__MODULE__{} = vectorizer, corpus) do
    vectorizer = fit(vectorizer, corpus)
    {vectorizer, transform(vectorizer, corpus)}
  end
end
