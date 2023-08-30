defmodule Mighty.Preprocessing.CountVectorizer do
  @moduledoc ~S"""
  A `CountVectorizer` contains performs a `transform/2` function that converts a corpus into a
  term frequency (TF) matrix and a document frequency (DF) matrix.

  The term frequency matrix is a matrix of shape $\lvert$ corpus $\rvert \times \lvert$ vocabulary $\rvert$ where each row
  is a document and each column is a feature. It represents the count of each feature in each document.
  The document frequency matrix is a vector of length $\lvert$ vocabulary $\rvert$ where each element is the
  number of documents that contain the feature at the corresponding index in the vocabulary divided by the
  number of documents in the corpus (hence the term 'frequency').

  ## Creating a `CountVectorizer`

  There are a number of different options that can be used to control the vocabulary that is built from the corpus
  as well as control the features that are included in the term frequency matrix. You can provide your own vocabulary
  or you can let the `CountVectorizer` build the vocabulary from the corpus. If you provide your own vocabulary, you
  must pass the vocabulary as a list, MapSet, or Map. If you pass a list or MapSet, they must contain strings that
  represent the desired tokens in your vocabulary. If you provide a Map, the keys must be strings that represent the
  desired tokens in your vocabulary and the values must be integers that represent the index of the token in the
  vocabulary. The indices must be consecutive integers from 0..length(vocabulary). If you do not provide your own vocabulary,
  the `CountVectorizer` will build the vocabulary from the corpus using the `preprocessor` and `tokenizer` options. The
  `preprocessor` option is a function that takes a string and returns a string. The `tokenizer` option is a function
  that takes a string and returns a list of strings. The `preprocessor` function is applied to each document in the corpus
  before the `tokenizer` function is applied to each document in the corpus. The `tokenizer` function is applied to each
  document in the corpus before the ngrams are generated. The `ngram_range` option controls the ngrams that are generated
  from the tokens. The `ngram_range` option must be a tuple of integers where the first integer is the minimum ngram length
  and the second integer is the maximum ngram length. All ngrams between your max and min ngram range will be generated,
  meaning `ngram_range: {1,1}` will generate only unigrams, `ngram_range: {2,2}` will only generate bigrams, and
  `ngram_range: {1,2}` will generate both unigrams and bigrams. The `stop_words` option is a list of strings that represent tokens
  that should be removed from the corpus before the vocabulary is built. The `binary` option controls whether the term frequency
  matrix is binary or not. If `binary` is `true`, all non-zero counts are set to 1. This is useful for discrete probabilistic
  models that model binary events rather than integer counts.

  Control the number of features that are included in the term frequency matrix by setting the `max_features` option.
  Control the minimum and maximum document frequency of the features that are included in the term frequency matrix by setting
  the `min_df` and `max_df` options. Control the ngram range of the features that are included in the term frequency
  matrix by setting the `ngram_range` option. Control whether the term frequency matrix is binary or not
  by setting the `binary` option.

  ## Examples

  ```elixir
  iex> corpus = [
  ...>   "This is the first document",
  ...>   "This document is the second document",
  ...>   "And this is the third one",
  ...>   "Is this the first document"
  ...> ]
  iex> vectorizer = Mighty.Preprocessing.CountVectorizer.new(corpus)
  iex> tf = Mighty.Preprocessing.CountVectorizer.transform(vectorizer, corpus)
  iex> tf |> Nx.to_list()
  [
    [0, 1, 1, 1, 0, 0, 1, 0, 1],
    [0, 2, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 1, 0, 1]
  ]
  iex> vectorizer = Mighty.Preprocessing.CountVectorizer.new(corpus, ngram_range: {2, 2})
  iex> tf = Mighty.Preprocessing.CountVectorizer.transform(vectorizer, corpus)
  iex> tf |> Nx.to_list()
  [
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]
  ]
  iex> vectorizer = Mighty.Preprocessing.CountVectorizer.new(corpus, max_features: 5, ngram_range: {1, 2}, min_df: 2, max_df: 0.8)
  iex> tf = Mighty.Preprocessing.CountVectorizer.transform(vectorizer, corpus)
  iex> tf |> Nx.to_list()
  [
    [1, 1],
    [2, 1],
    [0, 1],
    [1, 0]
  ]
  ```
  """
  defstruct vocabulary: nil,
            ngram_range: {1, 1},
            max_features: nil,
            min_df: 1,
            max_df: 1.0,
            stop_words: [],
            binary: false,
            preprocessor: nil,
            tokenizer: nil

  defp make_ngrams(tokens, ngram_range) do
    {min_n, max_n} = ngram_range
    n_original_tokens = length(tokens)

    ngrams =
      for n <- min_n..min(max_n, n_original_tokens) do
        for i <- 0..(n_original_tokens - n) do
          Enum.slice(tokens, i, n) |> Enum.join(" ")
        end
      end

    ngrams |> Enum.flat_map(& &1)
  end

  defp do_process(vectorizer = %__MODULE__{}, doc) do
    {pre_mod, pre_func, pre_args} = vectorizer.preprocessor
    {token_mod, token_func, token_args} = vectorizer.tokenizer

    doc
    |> then(fn doc -> apply(pre_mod, pre_func, [doc | pre_args]) end)
    |> then(fn doc -> apply(token_mod, token_func, [doc | token_args]) end)
    |> make_ngrams(vectorizer.ngram_range)
    |> Enum.filter(fn token ->
      if not is_nil(vectorizer.stop_words), do: token not in vectorizer.stop_words, else: true
    end)
  end

  defp build_vocab(vectorizer, corpus) do
    vocabulary =
      case vectorizer.vocabulary do
        nil ->
          corpus
          |> Enum.reduce([], fn doc, vocab ->
            vocab ++ do_process(vectorizer, doc)
          end)
          |> Enum.uniq()
          |> Enum.sort()
          |> Enum.with_index()
          |> Enum.into(%{})

        _ ->
          vectorizer.vocabulary
      end

    struct(vectorizer, vocabulary: vocabulary)
  end

  @doc """
  Creates a new CountVectorizer.
  Fits the vocabulary on the corpus if no vocabulary is provided.
  """
  def new(corpus, opts \\ []) do
    opts = Mighty.Preprocessing.Shared.validate_shared!(opts)
    # TODO: Any opts that are not needed for building the vocab should be moved from
    # new/2 to transform/2. Should those be stored in the struct?
    vectorizer = struct(__MODULE__, opts)
    build_vocab(vectorizer, corpus)
  end

  @doc """
  Transforms a corpus into a term frequency matrix and a document frequency matrix.

  The term frequency matrix is a matrix of shape (length(corpus), length(vocabulary) where each row
  is a document and each column is a feature. It represents the count of each feature in each document.

  The document frequency matrix is a vector of length (length(vocabulary)) where each element is the
  number of documents that contain the feature at the corresponding index in the vocabulary.
  """
  def transform(vectorizer = %__MODULE__{}, corpus) do
    idx_updates =
      corpus
      |> Enum.with_index()
      |> Enum.reduce([], fn {doc, doc_idx}, accum ->
        counts =
          doc
          |> then(&do_process(vectorizer, &1))
          |> Enum.reduce(
            Enum.map(vectorizer.vocabulary, fn {k, _} -> {k, 0} end) |> Enum.into(%{}),
            fn token, acc ->
              case Map.get(acc, token) do
                nil ->
                  Map.put(acc, token, 1)

                count ->
                  Map.put(acc, token, count + 1)
              end
            end
          )
          # Filter OOV tokens
          |> Enum.filter(fn {k, _v} -> not is_nil(Map.get(vectorizer.vocabulary, k)) end)
          |> Enum.map(fn {k, v} -> [doc_idx, Map.get(vectorizer.vocabulary, k), v] end)

        counts ++ accum
      end)
      |> Enum.reverse()
      |> Nx.tensor()

    tf =
      Nx.broadcast(0, {length(corpus), Enum.count(vectorizer.vocabulary)})
      |> Nx.indexed_put(idx_updates[[.., 0..1]], idx_updates[[.., 2]])

    tf =
      case vectorizer.max_features do
        nil ->
          tf

        max_features ->
          tf
          |> Nx.sum(axes: [0])
          |> Nx.argsort(axis: 0, direction: :desc)
          |> then(&Nx.take(tf, &1, axis: 1))
          |> Nx.slice_along_axis(0, max_features, axis: 1)
      end

    tf =
      if vectorizer.binary do
        Nx.select(Nx.greater(tf, 0), 1, 0)
      else
        tf
      end

    df = Scholar.Preprocessing.binarize(tf)

    # When max_df or min_df is a float, it is interpreted as a proportion of the total number of documents,
    # so we use the mean of the document frequencies to compare against the max_df or min_df.

    # When max_df or min_df is an integer, it is interpreted as an absolute count, so we use the sum of the
    # document frequencies to compare against the max_df or min_df.
    max_cond =
      case vectorizer.max_df do
        max_df when is_integer(max_df) ->
          Nx.less_equal(Nx.sum(df, axes: [0]), max_df)

        max_df when is_float(max_df) ->
          Nx.less_equal(Nx.mean(df, axes: [0]), max_df)

        _ ->
          raise ArgumentError, "max_df must be an integer or float in the range [0.0, 1.0]"
      end

    min_cond =
      case vectorizer.min_df do
        min_df when is_integer(min_df) ->
          Nx.greater_equal(Nx.sum(df, axes: [0]), min_df)

        min_df when is_float(min_df) ->
          Nx.greater_equal(Nx.mean(df, axes: [0]), min_df)

        _ ->
          raise ArgumentError, "min_df must be an integer or float in the range [0.0, 1.0]"
      end

    true_values =
      Nx.logical_and(
        min_cond,
        max_cond
      )

    true_indices = Nx.argsort(true_values, axis: 0, direction: :desc)

    true_count = Nx.sum(true_values) |> Nx.to_number()

    tf =
      Nx.take(tf, true_indices, axis: 1)
      |> Nx.slice_along_axis(0, true_count, axis: 1)

    tf
  end
end
