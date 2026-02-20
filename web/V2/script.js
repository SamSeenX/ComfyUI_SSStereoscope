lucide.createIcons();

function copyCode() {
  const codeText = document.getElementById("code-block").innerText;

  // Create a temporary textarea to hold the text
  const textarea = document.createElement("textarea");
  textarea.value = codeText;
  document.body.appendChild(textarea);
  textarea.select();

  try {
    // Execute the copy command
    document.execCommand("copy");
    alert("Installation commands copied to clipboard!");
  } catch (err) {
    console.error("Failed to copy text: ", err);
  } finally {
    // Clean up the temporary textarea
    document.body.removeChild(textarea);
  }
}
